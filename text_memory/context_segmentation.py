"""Confirmed whole-session partition selection, promoted without tuning.

Source: outputs/session-boundary-probe-20261025/multi_session.py.
The completion temperature uses half the permutation SD, as in the final
context-refinement run (diagnostics["0.5"]).
"""
import hashlib
import random
import statistics

SEED = 20261025
PERMUTATIONS = 200

def fit(kernel):
    """Return exact minimum within-segment SSE and boundaries for every k."""
    n = len(kernel)
    prefix = [[0.]*(n+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(n):
            prefix[i+1][j+1] = kernel[i][j]+prefix[i][j+1]+prefix[i+1][j]-prefix[i][j]
    diag = [0.]
    for i in range(n):
        diag.append(diag[-1]+kernel[i][i])
    costs = [[0.]*(n+1) for _ in range(n)]
    for a in range(n):
        for b in range(a+1, n+1):
            total = prefix[b][b]-prefix[a][b]-prefix[b][a]+prefix[a][a]
            costs[a][b] = max(0., diag[b]-diag[a]-total/(b-a))
    dp = [[float('inf')]*(n+1) for _ in range(n+1)]
    prev = [[None]*(n+1) for _ in range(n+1)]
    dp[0][0] = 0.
    for k in range(1, n+1):
        for b in range(k, n+1):
            dp[k][b], prev[k][b] = min((dp[k-1][a]+costs[a][b], a) for a in range(k-1, b))
    fits = []
    for k in range(1, n+1):
        blocks, j, b = [], k, n
        while j:
            a = prev[j][b]
            blocks.append([a+1, b])
            b, j = a, j-1
        fits.append({'k': k, 'sse': dp[k][n], 'blocks': blocks[::-1]})
    return fits


def choose_penalty(fits, beta):
    return min(fits, key=lambda f: (f['sse']+beta*(f['k']-1), f['k']))


def adaptive(fits, kernel, rng):
    n = len(fits)
    # A geometric knee: largest distance below the endpoint chord after scaling
    # both axes to [0,1]. No hand-supplied beta, though this criterion is a choice.
    total = fits[0]['sse']
    knee = [1-f['sse']/total-(f['k']-1)/(n-1) for f in fits]
    k_elbow = max(range(n), key=lambda j: (knee[j], -j))
    null = []
    for _ in range(PERMUTATIONS):
        order = list(range(n))
        rng.shuffle(order)
        shuffled = [[kernel[i][j] for j in order] for i in order]
        null.append([f['sse'] for f in fit(shuffled)])
    means = [statistics.mean(x[k] for x in null) for k in range(n)]
    deviations = [statistics.stdev(x[k] for x in null) for k in range(n)]
    gaps = [means[k]-fits[k]['sse'] for k in range(n)]
    k_gap = max(range(n), key=lambda j: (gaps[j], -j))
    # Deliberately conservative heuristic: prefer the smallest k whose gap is
    # within one null SD of the maximum. This is not a statistical confidence
    # interval and SD is NOT divided by sqrt(number of Monte Carlo draws).
    cutoff = gaps[k_gap]-deviations[k_gap]
    k_conservative = next(k for k in range(n) if gaps[k] >= cutoff-1e-12)
    return dict(elbow=fits[k_elbow], permutation_gap=fits[k_gap],
                permutation_gap_conservative=fits[k_conservative]), dict(
                    knee_scores=knee, permutation_mean_sse=means,
                    permutation_sd_sse=deviations, permutation_gaps=gaps,
                    conservative_gap_cutoff=cutoff)


def prepare_segments(kernel, request_id):
    fits = fit(kernel)
    if not fits:
        raise ValueError("a session must contain facts")
    if len(fits) == 1 or fits[0]["sse"] == 0.:
        # A single fact or identical vectors has no meaningful cut location.
        core, deviation = fits[0], 0.
    else:
        local_seed = SEED + int(hashlib.sha256(request_id.encode()).hexdigest()[:12], 16)
        choices, diagnostics = adaptive(fits, kernel, random.Random(local_seed))
        core = choices["permutation_gap_conservative"]
        deviation = diagnostics["permutation_sd_sse"][core["k"] - 1]
    return {"blocks": core["blocks"], "k": core["k"],
            "temperature": max(deviation, 1e-12) * .5}
