# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 19:23:03 2023

@author: Cristobal
"""
import numpy as np
import time
import random
def hfun(v0, rho):
    t0, s0, r0 = v0
    exprho = np.exp(rho)
    expnegrho = np.exp(-rho)
    f = ((rho - 1) * r0 + s0) * \
        exprho - (r0 - rho * s0) * \
            expnegrho - (rho * (rho - 1) + 1) * t0
    df = (rho * r0 + s0) * \
        exprho + (r0 - (rho - 1) * s0) * expnegrho - (2 * rho - 1) * t0
    return [f, df]

def rootsearch_bn(fun, farg, xl, xh, x0):
    EPS = np.finfo(np.float64).eps

    assert xl <= x0 <= xh

    xx = None

    while True:
        f = fun(farg, x0)[0]
        if f < 0.0:
            xl = x0
        else:
            xh = x0

        xx = 0.5 * (xl + xh)

        if abs(xx - x0) <= EPS * max(1., abs(xx)) or xx == xl or xx == xh:
            break

        x0 = xx

    return xx
# Newton root search for strictly increasing functions
def rootsearch_ntinc(fun, farg, xl, xh, x0):
    EPS = np.finfo(np.float64).eps
    DFTOL = EPS ** (6 / 7)
    MAXITER = 20
    LODAMP = float("0.05")
    HIDAMP = float("0.95")

    assert xl <= x0 <= xh

    xx = x0
    converged = False

    for i in range(MAXITER):
        f, df = fun(farg, x0)
        if f < 0.0:
            xl = x0
        else:
            xh = x0

        if xh <= xl:
            converged = True
            break

        if np.isfinite(f) and df >= DFTOL:
            xx = x0 - f / df
        else:
            break

        if abs(xx - x0) <= EPS * max(1., abs(xx)):
            converged = True
            break
        # Dampened steps to boundary
        if xx >= xh:
            x0 = min(LODAMP * x0 + HIDAMP * xh, xh)
        elif xx <= xl:
            x0 = max(LODAMP * x0 + HIDAMP * xl, xl)
        else:
            x0 = xx

        #print(["NT", x0, xl, xh, f])

    if converged:
        return max(xl, min(xh, xx))
    else:
        return rootsearch_bn(fun, farg, xl, xh, x0)

def projheu_primalexpcone(v0):
    t0, s0, r0 = v0

    # perspective boundary
    vp = np.array([max(t0, 0), 0.0, min(r0, 0)])
    dist = np.linalg.norm(np.array(vp) - np.array(v0), 2)

    # perspective interior
    if s0 > 0.0:
        tp = max(t0, s0 * np.exp(r0 / s0))
        newdist = tp - t0
        if newdist < dist:
            vp = np.array([tp, s0, r0])
            dist = newdist

    return [vp, dist]

def projheu_polarexpcone(v0):
    t0, s0, r0 = v0

    # perspective boundary
    vd = np.array([min(t0, 0), min(s0, 0), 0.0])
    dist = np.linalg.norm(np.array(vd) - np.array(v0), 2)

    # perspective interior
    if r0 > 0.0:
        td = min(t0, -r0 * np.exp(s0 / r0 - 1))
        newdist = t0 - td
        if newdist < dist:
            vd = np.array([td, s0, r0])
            dist = newdist

    return [vd, dist]

def projsol_primalexpcone(v0, rho):
    t0, s0, r0 = v0
    vp, dist = None, None

    linrho = ((rho - 1) * r0 + s0)
    exprho = np.exp(rho)
    if (linrho > 0) and np.isfinite(exprho):
        quadrho = rho * (rho - 1) + 1
        vp = np.array([exprho, 1, rho]) * linrho / quadrho
        dist = np.linalg.norm(np.array(vp) - np.array(v0), 2)
    else:
        vp = np.array([np.inf, 0.0, 0.0])
        dist = np.inf

    return [vp, dist]

def projsol_polarexpcone(v0, rho):
    t0, s0, r0 = v0
    vd, dist = None, None

    linrho = (r0 - rho * s0)
    exprho = np.exp(-rho)
    if (linrho > 0) and np.isfinite(exprho):
        quadrho = rho * (rho - 1) + 1
        vd = np.array([-exprho, 1 - rho, 1]) * linrho / quadrho
        dist = np.linalg.norm(np.array(vd) - np.array(v0), 2)
    else:
        vd = np.array([-np.inf, 0.0, 0.0])
        dist = np.inf

    return [vd, dist]

def ppsi(v0):
    t0, s0, r0 = v0
    
    # Two expressions for the same to avoid catastrophic cancellation
    if r0 > s0:
        psi = (r0 - s0 + np.sqrt(r0 ** 2 + s0 ** 2 - r0 * s0)) / r0
    else:
        psi = -s0 / (r0 - s0 - np.sqrt(r0 ** 2 + s0 ** 2 - r0 * s0))
    
    return ((psi - 1) * r0 + s0) / (psi * (psi - 1) + 1)

def pomega(rho):
    val = np.exp(rho) / (rho * (rho - 1) + 1)
    if rho < 2.0:
        val = min(val, np.exp(2) / 3)
   
    return val

def dpsi(v0):
    t0, s0, r0 = v0
    
    # Two expressions for the same to avoid catastrophic cancellation
    if s0 > r0:
        psi = (r0 - np.sqrt(r0 ** 2 + s0 ** 2 - r0 * s0)) / s0
    else:
        psi = (r0 - s0) / (r0 + np.sqrt(r0 ** 2 + s0 ** 2 - r0 * s0))
    
    res = (r0 - psi * s0) / (psi * (psi - 1) + 1)
    return res

def domega(rho):
    val = -np.exp(-rho) / (rho * (rho - 1) + 1)
    if rho > -1.0:
        val = max(val, -np.exp(1) / 3)
   
    return val

def searchbracket(v0, pdist, ddist):
    t0, s0, r0 = v0
    baselow, baseupr = float('-inf'), float('inf')
    low, upr = float('-inf'), float('inf')

    Dp = np.sqrt(pdist ** 2 - min(s0, 0) ** 2)
    Dd = np.sqrt(ddist ** 2 - min(r0, 0) ** 2)

    if t0 > 0:
        tpl = t0
        curbnd = np.log(tpl / ppsi(v0))
        low = max(low, curbnd)

    if t0 < 0:
        tdu = t0
        curbnd = -np.log(-tdu / dpsi(v0))
        upr = min(upr, curbnd)

    if r0 > 0:
        baselow = 1 - s0 / r0
        low = max(low, baselow)

        tpu = max(1e-12, min(Dd, Dp + t0))
        palpha = low
        curbnd = max(palpha, baselow + tpu / r0 / pomega(palpha))
        upr = min(upr, curbnd)

    if s0 > 0:
        baseupr = r0 / s0
        upr = min(upr, baseupr)

        tdl = -max(1e-12, min(Dp, Dd - t0))
        dalpha = upr
        curbnd = min(dalpha, baseupr - tdl / s0 / domega(dalpha))
        low = max(low, curbnd)

    assert baselow <= baseupr
    assert np.isfinite(low)
    assert np.isfinite(upr)

    # Guarantee valid bracket
    low, upr = min(low, upr), max(low, upr)
    low, upr = np.clip(low, baselow, baseupr), np.clip(upr, baselow, baseupr)
    if low != upr:
        fl = hfun(v0, low)[0]
        fu = hfun(v0, upr)[0]

        if not (fl * fu < 0):
            if abs(fl) < abs(fu) or np.isnan(fl):
                upr = low
            else:
                low = upr

    return [low, upr]

def proj_primalexpcone(v0):
    TOL = np.finfo(float).eps ** (2 / 3)
    v0 = np.array(v0)  # Convert input to NumPy array
    t0, s0, r0 = v0

    vp, pdist = projheu_primalexpcone(v0)
    vd, ddist = projheu_polarexpcone(v0)
    # Skip root search if presolve rules apply
    # or optimality conditions are satisfied
    if not ((s0 <= 0 and r0 <= 0) or 
            min(pdist, ddist) <= TOL or 
            (np.linalg.norm(vp + vd - v0, np.inf) <= TOL and 
             np.dot(vp, vd) <= TOL)):
        xl, xh = searchbracket(v0, pdist, ddist)
        rho = rootsearch_ntinc(hfun, v0, xl, xh, 0.5 * (xl + xh))

        vp1, pdist1 = projsol_primalexpcone(v0, rho)
        if pdist1 <= pdist:
            vp, pdist = vp1, pdist1

        vd1, ddist1 = projsol_polarexpcone(v0, rho)
        if ddist1 <= ddist:
            vd, ddist = vd1, ddist1

    return [vp, vd]

def abserr(v0, vp, vd):
    return [np.linalg.norm(vp + vd - v0, 2), 
            np.abs(np.dot(vp, vd))]

def relerr(v0, vp, vd):
    return abserr(v0, vp, vd) / max(1.0, np.linalg.norm(v0, 2))

def solutionreport(v0, vp, vd):
    abs_err = abserr(v0, vp, vd)
    rel_err = relerr(v0, vp, vd)
    v0_str = np.array2string(np.array(v0, dtype=np.float64), precision=6, suppress_small=True)
    vp_str = np.array2string(np.array(vp, dtype=np.float64), precision=6, suppress_small=True)
    vd_str = np.array2string(np.array(vd, dtype=np.float64), precision=6, suppress_small=True)
    print(f"abserr={abs_err[0]:.1e}, {abs_err[1]:.1e}")
    print(f"relerr={rel_err[0]:.1e}, {rel_err[1]:.1e}")
    print(f"  v0={v0_str}")
    print(f"  vp={vp_str} in primal")
    print(f"  vd={vd_str} in polar")

def test(real):
    low = -20
    upr = 21
    domain = np.concatenate((-np.exp(np.arange(low, upr + 1)), np.array([0.0]), np.exp(np.arange(low, upr + 1))))

    max_err1 = 0.0
    max_err2 = 0.0

    # Precompile and extract stats
    try:
        for t0 in domain:
            for s0 in domain:
                for r0 in domain:
                    v0 = np.array([t0, s0, r0], dtype=real)
                    try:
                        vp, vd = proj_primalexpcone(v0)
                        cur_err1, cur_err2 = relerr(v0, vp, vd)
                        max_err1 = max(max_err1, cur_err1)
                        max_err2 = max(max_err2, cur_err2)
                    except Exception as e:
                        print(f"ERROR: {v0}")
                        raise e
    except:
        pass

    # Benchmark time
    num_proj = len(domain) ** 3
    time = 0.0
    for t0 in domain:
        for s0 in domain:
            for r0 in domain:
                v0 = np.array([t0, s0, r0], dtype=real)
                vp, vd = proj_primalexpcone(v0)
                time += 1

    return {
        "maxerr1": max_err1,
        "maxerr2": max_err2,
        "numproj": num_proj,
        "totaltime": time,
        "avgtime": time / num_proj
    }
#%%
# Test example
v0 = np.array([1, 1, 1], dtype=float)
vp, vd = proj_primalexpcone(v0)
solutionreport(v0, vp, vd)
#%%
N = 1000
n = 10**3
x = [random.randint(-n, n)*random.random() for i in range(0,N)]
eta = [random.randint(-n, n)*random.random() for i in range(0,N)]
delta = [random.randint(-n, n)*random.random() for i in range(0,N)] 
v0 = [np.array([x[i],eta[i],delta[i]]) for i in range(0,N)]
times = []
t_0 = time.time()
for i in range(0,N):
    tp = time.time()
    print(i)
    proj_primalexpcone(v0[i])
    tpf = time.time()
    times.append(tpf-tp)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
print("\naverage elapsed time =","{:.2e}".format(sum(times)/len(times)),
      "seconds")
