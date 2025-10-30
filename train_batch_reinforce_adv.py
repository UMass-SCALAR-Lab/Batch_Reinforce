from collections.abc import Callable
import json
import math
from pathlib import Path
import random
import re, ipdb, sys, os
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss, masked_mean
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
from rho_solver import solve_ab
from rho_cvx_solver import solve_importance_cvxpy

def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                # reward = 0.5
                reward = 0.1
            else:
                # reward = 0.01
                reward = 0.

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def main():
    seed = 42
    # wandb_project = None
    wandb_project = 'batch_reinforce'  # "tiny_grpo"
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # checkpoint_path = Path("./output")
    checkpoint_path = Path("/scratch3/workspace/ychittepu_umass_edu-eps/batch_reinforce/ours")
    checkpoint_interval = 50
    train_batch_size = 32
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    # Batch Reinforce params
    # f_divs in ['chi2', 'kl', 'reverse_kl', 'tv', 'hellinger', 'js']
    # kl used in TRPO for trust region
    f_type = 'kl'
    div_trust_region = 0.01
    run_name = 'batch-reinforce-grpo-style'

    # assert train_batch_size == rollouts_per_step, f"The Training batch should use the entire batch of trajectories. train_batch_size: f{train_batch_size} != rollouts_per_step: {rollouts_per_step} \n"
    # assert group_size == 1, f"Only one response expected per prompt for Batch Reinforce."

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )

    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project, name=run_name)

    var_zero_skip_update_cnt = 0
    cvxpy_fail_update_cnt = 0
    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        # ipdb.set_trace()

        with torch.no_grad():
            for q, a in zip(questions, answers):
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                print(
                    f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                log_probs = returns.clone()
                log_probs_ref = returns.clone()
                kl = returns.clone()
                # log_probs = sequences_log_probs(
                #     model=model,
                #     sequence_ids=sequence_ids,
                #     attention_mask=attention_mask,
                # )
                # log_probs_ref = sequences_log_probs(
                #     model=reference_model,
                #     sequence_ids=sequence_ids,
                #     attention_mask=attention_mask,
                # )
                # kl = approx_kl_divergence(
                #     log_probs=log_probs,
                #     log_probs_ref=log_probs_ref,
                #     action_mask=action_mask,
                # )

                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))

        # ipdb.set_trace()
        
        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            # This should only iterate once!!
            for i, exp in enumerate(experience_sampler):
                exp: Experience

                # if(i > 0):
                #     raise ValueError("Entire batch of samples should be used at once, not split into mini batches.")

                exp = exp.to(device)

                optimizer.zero_grad()

                # ipdb.set_trace()

                action_mask = exp.action_mask
                response_lengths = action_mask.sum(-1).detach().cpu()
                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                with torch.no_grad():
                    log_probs_ref = sequences_log_probs(
                        reference_model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                    )
                    kl = torch.sum((log_probs.detach().float() - log_probs_ref.float())*action_mask, dim=-1)

                # Look at advantages instead of returns
                batch_returns = exp.advantages.flatten()
                std = batch_returns.std(correction=0)

                if std.item() < 1e-12:
                    imp_ratios = torch.ones_like(batch_returns)
                    if torch.all(batch_returns <= 0):
                        # e.g., skip update or handle separately
                        var_zero_skip_update_cnt += 1
                        print(f"Skipping batch on account of all zero reward trajectories in batch")
                        continue
                else:
                    # This is chi-square solution without the KL wrt pi_ref consideration
                    # mean = batch_returns.mean()
                    # scale = math.sqrt(div_trust_region) / (std + 1e-8)
                    # imp_ratios = 1 + scale * (batch_returns - mean)

                    # This the the chi-square solution with the KL constraint
                    # Using Cvxpy
                    info = solve_importance_cvxpy(batch_returns.cpu().numpy(), beta=kl_weight, delta=div_trust_region, log_pi_old_over_ref=kl.cpu().numpy(), f_type=f_type)
                    if(info['status'] not in ("optimal", "optimal_inaccurate")):
                        cvxpy_fail_update_cnt += 1
                        print(f"Skipping batch on account of cvxpy optimization failure")
                        continue
                    imp_ratios = torch.from_numpy(info['rho']).to(device)

                    # # Own implementation
                    # batch_returns_aug = batch_returns - kl_weight*(1 + kl)
                    # batch_returns_aug_lst = batch_returns_aug.tolist()
                    # # a is lagrange multiplier for f-div constraint
                    # # b is lagrange multiplier for normalization constraint
                    # # rhos are solution to our optimization problem
                    # a, b, imp_ratios = solve_ab(batch_returns_aug_lst, kl_weight, div_trust_region)
                    # imp_ratios = torch.tensor(imp_ratios, device=device)

                # with torch.no_grad():
                #     log_probs_ref = sequences_log_probs(
                #         reference_model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                #     )

                # kl = approx_kl_divergence(
                #     log_probs=log_probs,
                #     log_probs_ref=log_probs_ref,
                #     action_mask=action_mask,
                # )

                # loss = (-log_probs * imp_ratios.view(-1,1)) + kl_weight * kl
                # (loss * action_mask).sum(axis=dim)

                # This is using the BC loss i.e forward KL i.e KL(pi* || pi)
                loss = -((log_probs*action_mask).sum(-1)*(imp_ratios-1)).mean()

                # This is using the Reverse KL projection loss i.e KL(pi || pi*)
                # loss = -((log_probs*action_mask).sum(-1)*torch.log(imp_ratios.detach()+1e-6)).mean()

                # loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    # print(f"experience.advantages={experience.advantages}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"{step_epoch}, grad_norm={grad_norm: .4f}")
                wandb.log(
                    {"grad_norm": grad_norm, "var_zero_updates_skipped": var_zero_skip_update_cnt, "cvxpy_fail_updates_skipped": cvxpy_fail_update_cnt, 
                    "advantages_mean": batch_returns.detach().mean().item(), "advantages_min": batch_returns.detach().min().item(), "advantages_max": batch_returns.detach().max().item(),
                    "imp_ratios_mean": imp_ratios.cpu().mean().item(), "imp_ratios_min": imp_ratios.cpu().min().item(), "imp_ratios_max": imp_ratios.cpu().max().item(),
                    "kl": kl.detach().mean().item(), "log_probs_avg": (log_probs*action_mask).detach().sum(-1).mean().item(), "response_lengths_mean": response_lengths.float().mean().item()
                    })

                optimizer.step()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
