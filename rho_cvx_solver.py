import numpy as np
import cvxpy as cp

import cvxpy as cp
import numpy as np

def f_div_constraint(rho, B, delta, f_type="chi2", eps=1e-8):
    """
    Return a list of CVXPY constraints that enforce D_f(p||q) <= delta
    where q is uniform and rho_i = p_i / q_i, so D_f = (1/B) * sum f(rho_i).

    Parameters
    ----------
    rho : cp.Variable (shape n,)  # nonnegative weights (p_i/q_i)
    B   : float or int            # sum(rho) == B  (i.e., mean of rho is 1)
    delta : float                 # divergence budget
    f_type : str                  # one of: 'chi2', 'kl', 'reverse_kl', 'tv', 'hellinger', 'js'
    eps : float                   # small lower bound to avoid log(0)

    Returns
    -------
    constraints : list[cp.Constraint]
    """
    constraints = [cp.sum(rho) == float(B)]  
    # domain: rho > 0 for log/sqrt

    if f_type.lower() in ("chi2", "chi-squared", "chi2-div"):
        # f(t) = (t - 1)^2
        f_vals = cp.square(rho - 1.0)
        constraints += [(1.0 / B) * cp.sum(f_vals) <= float(delta)]

    elif f_type.lower() in ("kl", "kullback-leibler", "relative_entropy", "rel_entr"):
        # f(t) = t*log(t) - t + 1
        f_vals = cp.rel_entr(rho, 1.0) - rho + 1.0
        constraints += [(1.0 / B) * cp.sum(f_vals) <= float(delta)]
        constraints += [rho >= eps]

    elif f_type.lower() in ("reverse_kl", "rkl", "kl_rev"):
        # f(t) = -log(t) + t - 1
        f_vals = -cp.log(rho) + rho - 1.0
        constraints += [(1.0 / B) * cp.sum(f_vals) <= float(delta)]
        constraints += [rho >= eps]

    elif f_type.lower() in ("tv", "total_variation", "variational"):
        # f(t) = 0.5 * |t - 1|
        f_vals = 0.5 * cp.abs(rho - 1.0)
        constraints += [(1.0 / B) * cp.sum(f_vals) <= float(delta)]

    elif f_type.lower() in ("hellinger", "hellinger2", "squared_hellinger"):
        # f(t) = (sqrt(t) - 1)^2 = t - 2*sqrt(t) + 1
        f_vals = rho - 2.0 * cp.sqrt(rho) + 1.0
        constraints += [(1.0 / B) * cp.sum(f_vals) <= float(delta)]

    elif f_type.lower() in ("js", "jensen-shannon", "jsd"):
        # One valid convex form:
        # f(t) = t*log t - (t+1) * log((t+1)/2) + log 2
        # Use cvxpy's rel_entr to stay DCP-compliant:
        f_vals = cp.rel_entr(rho, 1.0) - cp.rel_entr(rho + 1.0, 2.0) + np.log(2.0)
        constraints += [(1.0 / B) * cp.sum(f_vals) <= float(delta)]
        constraints += [rho >= eps]

    else:
        raise ValueError(f"Unknown f-divergence type: {f_type}")

    return constraints


def solve_importance_cvxpy(
    g,
    beta,
    delta,
    log_pi_old_over_ref=None,   # elementwise log(pi_old/pi_ref); may be None,
    f_type = 'chi2',
    solver=None,                # if None, try ["ECOS", "SCS", "MOSEK"]
    verbose=False
):
    """
    max_{rho>0} (1/B) sum_i rho_i * ( G_i - beta*log(rho_i) - beta*log(pi_old_i/pi_ref_i) )
    s.t. (1/B) sum_i rho_i = 1
         (1/B) sum_i (rho_i - 1)^2 <= delta
    """
    # exact objective: rho * (G - beta*log rho - beta*log ratio)
    if log_pi_old_over_ref is None:
        const = g
    else:
        const = g - beta * log_pi_old_over_ref  # constant per i

    B = len(g)
    rho = cp.Variable(B, pos=True)

    # cp.entr(x) = -x*log(x)  =>  -beta * rho*log(rho)  == beta * entr(rho)
    objective = (1.0 / B) * cp.sum(rho * const + beta * cp.entr(rho))

    # constraints = [
    #     cp.sum(rho) == float(B),                       # mean = 1
    #     (1.0 / B) * cp.sum_squares(rho - 1.0) <= float(delta)  # variance <= delta
    # ]

    constraints = f_div_constraint(rho, B, delta, f_type=f_type, eps=1e-8)

    prob = cp.Problem(cp.Maximize(objective), constraints)

    solvers = [solver] if solver is not None else ["ECOS", "SCS", "CLARABEL"]
    tried = []
    for s in solvers:
        tried.append(s)
        try:
            prob.solve(solver=s, verbose=verbose)
        except Exception:
            continue
        if prob.status in ("optimal", "optimal_inaccurate"):
            r = rho.value
            return {
                "ok": True,
                "status": prob.status,
                "solver": s,
                "objective": float(prob.value) if prob.value is not None else None,
                "rho": r,
                "mean": float(np.mean(r)) if r is not None else None,
                "variance": float(np.mean((r - 1.0) ** 2)) if r is not None else None,
            }

    return {
        "ok": False,
        "status": f"failed: tried {tried} -> {prob.status if hasattr(prob,'status') else 'no_status'}",
        "solver": tried[-1] if tried else None,
        "objective": None,
        "rho": None,
        "mean": None,
        "variance": None,
    }

# -----------------------------
# Example usage
if __name__ == "__main__":
    g = np.array([0.2, 1.0, -0.5, 0.7], dtype=float)
    beta = 0.1
    delta = 1.0
    out = solve_importance_cvxpy(g, beta, delta, log_pi_old_over_ref=None, verbose=False)
    print("status:", out["status"])
    print("objective:", out["objective"])
    print("mean:", out["mean"])
    print("variance(attained):", out["variance"])
    print("rho:", out["rho"], type(out['rho']))
