"""Session-fact context windows from uncertainty over global segmentations.

No model or store access. Segment count and temperature are supplied by the
caller; support values are model weights, not calibrated relevance probabilities.
Indices in the public API are one-based inclusive, matching fact sequences.
"""
from math import exp, isfinite, log


def _logsum(values):
    values = list(values)
    peak = max(values, default=float('-inf'))
    if peak == float('-inf'):
        return peak
    return peak + log(sum(exp(v - peak) for v in values))


def segment_costs(kernel):
    n = len(kernel)
    if not n or any(len(row) != n for row in kernel):
        raise ValueError('kernel must be a nonempty square matrix')
    if any(not isfinite(v) for row in kernel for v in row):
        raise ValueError('kernel must be finite')
    if any(abs(kernel[i][j] - kernel[j][i]) > 1e-8 for i in range(n) for j in range(n)):
        raise ValueError('kernel must be symmetric')
    prefix = [[0.] * (n + 1) for _ in range(n + 1)]
    diag = [0.]
    for i in range(n):
        diag.append(diag[-1] + kernel[i][i])
        for j in range(n):
            prefix[i+1][j+1] = kernel[i][j] + prefix[i][j+1] + prefix[i+1][j] - prefix[i][j]
    costs = [[0.] * (n + 1) for _ in range(n)]
    for a in range(n):
        for b in range(a+1, n+1):
            total = prefix[b][b] - prefix[a][b] - prefix[b][a] + prefix[a][a]
            costs[a][b] = max(0., diag[b] - diag[a] - total / (b-a))
    return costs


def same_segment_support(kernel, k, temperature):
    """Exact forward/backward sum over all contiguous partitions with k parts.

    Weight(partition) is proportional to exp(-sum(segment SSE)/temperature).
    Conditioning on k deliberately isolates boundary location from count choice.
    """
    costs = segment_costs(kernel)
    n = len(kernel)
    if type(k) is not int or not 1 <= k <= n:
        raise ValueError('k must be between 1 and n')
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    neg = float('-inf')
    weights = [[-c / temperature for c in row] for row in costs]
    forward = [[neg] * (n+1) for _ in range(k+1)]
    backward = [[neg] * (n+1) for _ in range(k+1)]
    forward[0][0], backward[0][n] = 0., 0.
    for count in range(1, k+1):
        for b in range(count, n+1):
            forward[count][b] = _logsum(forward[count-1][a] + weights[a][b]
                                       for a in range(count-1, b))
        for a in range(n-count, -1, -1):
            backward[count][a] = _logsum(weights[a][b] + backward[count-1][b]
                                        for b in range(a+1, n-count+2))
    normalizer = forward[k][n]
    support = [[0.] * n for _ in range(n)]
    boundaries = [0.] * (n-1)
    intervals = []
    for a in range(n):
        for b in range(a+1, n+1):
            lp = _logsum(forward[left][a] + weights[a][b] + backward[k-left-1][b]
                         for left in range(k)) - normalizer
            probability = exp(min(0., lp))
            intervals.append({'start': a+1, 'end': b, 'support': probability})
            if b < n:
                boundaries[b-1] += probability
            for i in range(a, b):
                for j in range(a, b):
                    support[i][j] += probability
    return {'same_segment': [[min(1., v) for v in row] for row in support],
            'boundary': boundaries, 'intervals': intervals,
            'log_normalizer': normalizer, 'k': k, 'temperature': temperature}


def context_windows(blocks, support, *, minimum_support=.2):
    """One window per hit; preserve its entire original core, then add support.

    Never recursively expand through added facts or through overlapping windows.
    """
    n = len(support)
    if not n or any(len(row) != n for row in support):
        raise ValueError('support must be square and nonempty')
    if not 0 < minimum_support <= 1:
        raise ValueError('minimum_support must be in (0,1]')
    if any(not isfinite(v) or not 0 <= v <= 1 for row in support for v in row):
        raise ValueError('support must contain values in [0,1]')
    if any(type(a) is not int or type(b) is not int or not 1 <= a <= b <= n for a,b in blocks):
        raise ValueError('invalid core bounds')
    if [i for a,b in blocks for i in range(a,b+1)] != list(range(1,n+1)):
        raise ValueError('cores must partition the sequence in order')
    windows = []
    for hit in range(1,n+1):
        a,b = next((a,b) for a,b in blocks if a <= hit <= b)
        # Expand only through a continuous run at each edge.  Taking the hull
        # of every supported fact made one distant high-support fact pull all
        # intervening topics into the returned context.
        start, end = a, b
        while start > 1 and support[hit-1][start-2] >= minimum_support:
            start -= 1
        while end < n and support[hit-1][end] >= minimum_support:
            end += 1
        windows.append({'hit': hit, 'core': [a,b], 'context': [start,end],
                        'fact_numbers': list(range(start,end+1))})
    return windows
