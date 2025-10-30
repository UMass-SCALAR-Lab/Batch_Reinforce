# import mpmath as mp

# def rho_lambert(g, beta, a, b, eps=1e-14):
#     """Compute rho_i from the stationarity condition using Lambert W."""
#     if abs(a) < 1e-12:
#         return mp.e**((g - b)/beta)

#     z = (2*a/beta) * mp.e**((g - b + 2*a)/beta)
#     # Keep inside real-domain region for real W
#     z = max(z, -mp.e**(-1) + eps)

#     # Choose branch: principal for a>=0, lower branch for a<0
#     branch = 0 if a >= 0 else -1
#     w = mp.lambertw(z, branch)
#     # Guard: if w is complex due to numerical jitter, take real part if imag is tiny
#     if abs(mp.im(w)) > 1e-12:
#         raise RuntimeError(f"LambertW produced complex value (a={a}, b={b}, z={z}, branch={branch}).")
#     w = mp.re(w)
#     return (beta / (2*a)) * w


# def solve_b(a, g, beta, tol=1e-10, maxit=60):
#     """Solve for b given a by Newton iteration on mean constraint."""
#     B = len(g)
#     # start from a=0 softmax baseline
#     s = [mp.e**(gi/beta) for gi in g]
#     b = beta * mp.log(sum(s)/B)

#     for _ in range(maxit):
#         r = [rho_lambert(gi, beta, a, b) for gi in g]
#         M = sum(r)/B - 1
#         if abs(M) < tol:
#             return b

#         dM = -sum(1/(2*a + beta/ri) for ri in r)/B  # strictly negative
#         step = -M/dM
#         # Backtracking line-search for robustness
#         t = 1.0
#         improved = False
#         while t > 1e-6:
#             b_new = b + t*step
#             try:
#                 r_new = [rho_lambert(gi, beta, a, b_new) for gi in g]
#             except RuntimeError:
#                 t *= 0.5
#                 continue
#             M_new = sum(r_new)/B - 1
#             if abs(M_new) < abs(M):
#                 b = b_new
#                 improved = True
#                 break
#             t *= 0.5
#         if not improved:
#             # last resort small step
#             b += step * 1e-3
#     raise RuntimeError("solve_b(a): Newton failed to converge")


# def _variance_for_a(a, g, beta):
#     """Helper: compute variance and b(a); returns (V, b, r, ok)."""
#     try:
#         b = solve_b(a, g, beta)
#         r = [rho_lambert(gi, beta, a, b) for gi in g]
#         B = len(g)
#         V = sum((ri - 1)**2 for ri in r)/B
#         return V, b, r, True
#     except Exception:
#         return mp.nan, None, None, False


# def _coarse_scan_for_a(g, beta, delta, a_min, a_max, n=121):
#     """Scan a grid in [a_min, a_max] to find a sign change of Δ(a)."""
#     # Build grid (inclusive)
#     grid = [a_min + (a_max - a_min)*k/(n-1) for k in range(n)]
#     vals = []
#     for a in grid:
#         V, _, _, ok = _variance_for_a(a, g, beta)
#         if not ok:
#             vals.append(None)
#         else:
#             vals.append(V - delta)
#     # Find adjacent sign change ignoring None
#     last_idx = None
#     last_val = None
#     for i, v in enumerate(vals):
#         if v is None or mp.isnan(v):
#             continue
#         if last_val is not None and v*(last_val) <= 0:
#             # bracket found between grid[last_idx] and grid[i]
#             return True, (grid[last_idx], grid[i]), vals, grid
#         last_idx, last_val = i, v
#     return False, None, vals, grid


# def solve_ab(g_list, beta, delta, atol=1e-10, rtol=1e-8):
#     """
#     Solve for (a,b,{rho_i}) satisfying:
#       (1) mean constraint: (1/B) sum rho_i = 1
#       (2) variance constraint: (1/B) sum (rho_i-1)^2 = delta
#     Returns: (a, b, rhos)
#     """
#     B = len(g_list)
#     g = [mp.mpf(v) for v in g_list]
#     beta = mp.mpf(beta)
#     delta = mp.mpf(delta)

#     # ----- Trivial / early checks -----
#     if beta <= 0:
#         raise ValueError("beta must be positive.")
#     if B == 0:
#         raise ValueError("g_list must be non-empty.")

#     # All g equal?
#     if all(abs(gi - g[0]) < 1e-14 for gi in g):
#         if delta > 0:
#             raise RuntimeError("Infeasible: all g_i identical ⇒ only δ=0 attainable.")
#         return mp.mpf('0.0'), g[0], [mp.mpf('1.0')]*B

#     # Baseline a=0 softmax solution (variance δ0)
#     s = [mp.e**(gi/beta) for gi in g]
#     sbar = sum(s)/B
#     b0 = beta*mp.log(sbar)
#     rho0 = [si/sbar for si in s]
#     delta0 = sum((ri-1)**2 for ri in rho0)/B
#     if mp.almosteq(delta0, delta, rel_eps=rtol, abs_eps=atol):
#         return mp.mpf('0.0'), b0, rho0

#     # Helper to evaluate Δ(a)=V(a)-δ
#     def Delta(a):
#         V, _, _, ok = _variance_for_a(a, g, beta)
#         if not ok:
#             return mp.nan
#         return V - delta

#     # ----- Try to bracket around a=0 by geometric expansion -----
#     target_sign = mp.sign(delta - delta0)  # +1 wants larger variance than at a=0
#     aL, aR = mp.mpf('0.0'), mp.mpf('0.0')
#     step = mp.mpf('0.1')/beta
#     found = False
#     for _ in range(60):
#         aR += target_sign * step
#         val0 = Delta(mp.mpf('0.0'))  # = delta0 - delta
#         valR = Delta(aR)
#         if not (mp.isnan(val0) or mp.isnan(valR)):
#             if val0 * valR <= 0:
#                 aL = mp.mpf('0.0')
#                 found = True
#                 break
#         step *= 2

#     # ----- If not bracketed, do a coarse global scan -----
#     if not found:
#         # pick a wide symmetric range (scaled by beta for numerical sanity)
#         A = mp.mpf('1e3')/beta
#         ok, bracket, vals, grid = _coarse_scan_for_a(g, beta, delta, -A, A, n=241)
#         if ok:
#             aL, aR = bracket
#             found = True
#         else:
#             # Estimate attainable range for δ from the scan
#             finite_vals = [v for v in vals if v is not None and not mp.isnan(v)]
#             if len(finite_vals) == 0:
#                 raise RuntimeError("Could not evaluate Δ(a) reliably over the search range.")
#             # Δ = V - δ, so attainable V range = (Δ + δ)
#             V_vals = [v + delta for v in finite_vals]
#             Vmin, Vmax = min(V_vals), max(V_vals)
#             raise RuntimeError(
#                 f"Infeasible δ: requested δ={delta} outside attainable range "
#                 f"[{Vmin} , {Vmax}] (estimated)."
#             )

#     # ----- With a bracket, root-find a (Brent via secant-style mp.findroot) -----
#     f = lambda aa: Delta(aa)
#     try:
#         a = mp.findroot(f, (aL, aR))
#     except:  # fallback: simple secant iterations with damping
#         a_prev, a_curr = aL, aR
#         f_prev, f_curr = f(a_prev), f(a_curr)
#         for _ in range(60):
#             if abs(f_curr - f_prev) < 1e-18:
#                 break
#             a_next = a_curr - f_curr*(a_curr - a_prev)/(f_curr - f_prev)
#             # damping if we go nan
#             damp = 1.0
#             val_next = Delta(a_next)
#             while mp.isnan(val_next) and damp > 1e-6:
#                 a_next = a_curr + 0.5*damp*(a_next - a_curr)
#                 val_next = Delta(a_next)
#                 damp *= 0.5
#             if mp.isnan(val_next):
#                 break
#             a_prev, f_prev = a_curr, f_curr
#             a_curr, f_curr = a_next, val_next
#             if abs(f_curr) < 1e-10:
#                 break
#         a = a_curr

#     # Final evaluate b and rhos
#     b = solve_b(a, g, beta)
#     rhos = [rho_lambert(gi, beta, a, b) for gi in g]
#     # Small cleanup: ensure mean≈1
#     B = len(g)
#     mean_r = sum(rhos)/B
#     if abs(mean_r - 1) > 1e-7:
#         # Tiny renorm (should be unnecessary, but safe)
#         rhos = [ri/mean_r for ri in rhos]
#     return a, b, rhos


import mpmath as mp

mp.mp.dps = 80  # precision

def rho_lambert(g, beta, a, b, eps=mp.mpf('1e-30')):
    """Compute rho_i from the stationarity condition using Lambert W."""
    if abs(a) < mp.mpf('1e-20'):
        return mp.e**((g - b)/beta)

    z = (2*a/beta) * mp.e**((g - b + 2*a)/beta)
    z_floor = -mp.e**(-1) + eps
    if z < z_floor:
        z = z_floor

    branch = 0 if a >= 0 else -1  # a<0 -> W_{-1} to keep rho>0
    w = mp.lambertw(z, branch)
    if abs(mp.im(w)) > mp.mpf('1e-40'):
        raise RuntimeError(f"LambertW produced complex value (a={a}, b={b}, z={z}, branch={branch}).")
    w = mp.re(w)
    return (beta / (2*a)) * w


def solve_b(a, g, beta, tol=mp.mpf('1e-20'), maxit=80):
    """Solve for b given a by Newton iteration on mean constraint."""
    B = len(g)
    s = [mp.e**(gi/beta) for gi in g]
    b = beta * mp.log(sum(s)/B)  # start at a=0 softmax b

    for _ in range(maxit):
        r = [rho_lambert(gi, beta, a, b) for gi in g]
        M = sum(r)/B - 1
        if abs(M) < tol:
            return b

        dM = -sum(1/(2*a + beta/ri) for ri in r)/B  # < 0
        step = -M/dM

        # Backtracking line search
        t = mp.mpf('1.0')
        improved = False
        while t > mp.mpf('1e-12'):
            b_new = b + t*step
            try:
                r_new = [rho_lambert(gi, beta, a, b_new) for gi in g]
            except RuntimeError:
                t *= mp.mpf('0.5')
                continue
            M_new = sum(r_new)/B - 1
            if abs(M_new) < abs(M):
                b = b_new
                improved = True
                break
            t *= mp.mpf('0.5')
        if not improved:
            b += step * mp.mpf('1e-6')  # tiny nudge
    raise RuntimeError("solve_b(a): Newton failed to converge")


def _variance_for_a(a, g, beta):
    """Helper: compute variance and b(a); returns (V, b, r, ok)."""
    try:
        b = solve_b(a, g, beta)
        r = [rho_lambert(gi, beta, a, b) for gi in g]
        B = len(g)
        V = sum((ri - 1)**2 for ri in r)/B
        return V, b, r, True
    except Exception:
        return mp.nan, None, None, False


def _coarse_scan_for_a(g, beta, delta, a_min, a_max, n=201):
    """Scan a grid in [a_min, a_max] to find a sign change of Δ(a)."""
    grid = [a_min + (a_max - a_min)*k/(n-1) for k in range(n)]
    vals = []
    for a in grid:
        V, _, _, ok = _variance_for_a(a, g, beta)
        vals.append((V - delta) if ok else None)
    last_idx = None
    last_val = None
    for i, v in enumerate(vals):
        if v is None or mp.isnan(v):
            continue
        if last_val is not None and v*last_val <= 0:
            return True, (grid[last_idx], grid[i]), vals, grid
        last_idx, last_val = i, v
    return False, None, vals, grid


def solve_ab(g_list, beta, delta,
             atol=mp.mpf('1e-14'), rtol=mp.mpf('1e-10'),
             variance_floor=mp.mpf('1e-30'),
             max_a=mp.mpf('1e6')):
    """
    Solve for (a,b,{rho_i}) satisfying:
      (1) mean constraint: (1/B) sum rho_i = 1
      (2) variance constraint: (1/B) sum (rho_i-1)^2 = delta
    Returns: (a, b, rhos)
    """
    B = len(g_list)
    if B == 0:
        raise ValueError("g_list must be non-empty.")
    beta = mp.mpf(beta)
    if beta <= 0:
        raise ValueError("beta must be positive.")
    delta = mp.mpf(delta)
    g = [mp.mpf(v) for v in g_list]

    # All g equal?
    if all(abs(gi - g[0]) < mp.mpf('1e-30') for gi in g):
        if delta > variance_floor:
            raise RuntimeError("Infeasible: all g_i identical ⇒ only δ≈0 attainable.")
        return mp.mpf('0.0'), g[0], [mp.mpf('1.0')]*B

    # Baseline a=0 softmax
    s = [mp.e**(gi/beta) for gi in g]
    sbar = sum(s)/B
    b0 = beta*mp.log(sbar)
    rho0 = [si/sbar for si in s]
    delta0 = sum((ri-1)**2 for ri in rho0)/B
    if abs(delta0 - delta) <= max(rtol*max(delta, mp.mpf('1.0')), atol):
        return mp.mpf('0.0'), b0, rho0

    # Δ(a)=V(a)-δ with a small floor to avoid chasing a→+∞
    def Delta(a):
        V, _, _, ok = _variance_for_a(a, g, beta)
        if not ok:
            return mp.nan
        if V < variance_floor:
            V = mp.mpf('0.0')
        return V - delta

    # Direction fix: if δ > δ0 (need larger variance), move a NEGATIVE.
    direction = -1 if delta > delta0 else +1

    # Try to bracket around a=0
    aL, aR = mp.mpf('0.0'), mp.mpf('0.0')
    step = mp.mpf('0.1')/max(beta, mp.mpf('1e-12'))
    val0 = Delta(mp.mpf('0.0'))  # = delta0 - delta
    found = False
    for _ in range(60):
        aR += direction * step
        if abs(aR) > max_a:
            break
        valR = Delta(aR)
        if not (mp.isnan(val0) or mp.isnan(valR)):
            if val0 * valR <= 0:
                aL = mp.mpf('0.0')
                found = True
                break
        step *= 2

    # If no local bracket, coarse scan in [-max_a, max_a]
    if not found:
        ok, bracket, vals, grid = _coarse_scan_for_a(g, beta, delta, -max_a, max_a, n=321)
        if ok:
            aL, aR = bracket
            found = True
        else:
            finite_vals = [v for v in vals if v is not None and not mp.isnan(v)]
            if len(finite_vals) == 0:
                raise RuntimeError("Could not evaluate Δ(a) reliably over the search range.")
            V_vals = [v + delta for v in finite_vals]
            Vmin, Vmax = min(V_vals), max(V_vals)
            raise RuntimeError(
                f"Infeasible δ: requested δ={delta} outside attainable range "
                f"[{Vmin} , {Vmax}] (estimated)."
            )

    # Root-find a within the bracket
    f = lambda aa: Delta(aa)
    try:
        a = mp.findroot(f, (aL, aR))
    except Exception:
        # damped secant fallback
        a_prev, a_curr = aL, aR
        f_prev, f_curr = f(a_prev), f(a_curr)
        for _ in range(100):
            if abs(f_curr - f_prev) < mp.mpf('1e-40'):
                break
            a_next = a_curr - f_curr*(a_curr - a_prev)/(f_curr - f_prev)
            a_next = max(-max_a, min(max_a, a_next))
            val_next = f(a_next)
            if mp.isnan(val_next):
                a_next = (a_curr + a_prev)/2
                val_next = f(a_next)
            a_prev, f_prev = a_curr, f_curr
            a_curr, f_curr = a_next, val_next
            if abs(f_curr) < mp.mpf('1e-18'):
                break
        a = a_curr

    # Final b and rhos
    b = solve_b(a, g, beta)
    rhos = [rho_lambert(gi, beta, a, b) for gi in g]
    mean_r = sum(rhos)/B
    if mean_r != 0:
        rhos = [ri/mean_r for ri in rhos]  # tiny renorm
    return a, b, rhos
