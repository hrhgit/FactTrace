"""Session-fact context assembly: local overlap plus non-recursive distant packets.

Lexical/semantic matches are retrieval hints, never assertions of object identity.
Fact indices are one-based. Input sequence must belong to one conversation.
"""
from collections import Counter
from math import isfinite

from .context_windows import context_windows
from .collections.retrieval import REPORTING_TERMS
from .retrieval import STOPWORDS
from .store import terms

PRONOUNS = {"w" + word for word in
            "i me my mine we us our ours you your yours he him his she her hers they them their theirs it its this that these those who whom someone something".split()}
GENERIC_LINK_WORDS = {"w" + word for word in
                      "during conversation time take way get got go went make made said told asked shared found good great new one first last would could also really very like love feels feel people work things day".split()}


def complete_context(blocks, support, kernel, texts, *, participants=(), speakers=(),
                     minimum_support=.4, link_floor=.66, bridge_floor=.55, companion_support=.7,
                     companion_cosine=.68):
    """Attach strong direct cross-core matches without filling intervening facts.

    The objective is lexicographic: preserve the required context first, then
    reduce unrelated facts. Thresholds therefore err toward recall; returned
    size is only a secondary criterion.

    An anchor needs two distinct content terms, including one present in fewer
    than half of session facts, plus semantic agreement. A companion must be in
    that anchor's original core, be from the same speaker when speaker labels
    are supplied, and be supported by its local segmentation and vector.
    Only the original hit seeds links; added facts never seed further packets.
    """
    windows = context_windows(blocks, support, minimum_support=minimum_support)
    n=len(texts)
    if n != len(support) or len(kernel)!=n or any(len(row)!=n for row in kernel):
        raise ValueError('texts, kernel and support sizes must match')
    if any(not isinstance(t,str) or not t.strip() for t in texts):
        raise ValueError('texts must be nonempty strings')
    if speakers and len(speakers) != n:
        raise ValueError('speakers must match texts when supplied')
    if any(not isfinite(v) or not -1.000001<=v<=1.000001 for row in kernel for v in row):
        raise ValueError('kernel must contain finite cosine values')
    if not -1<=link_floor<=1 or not 0<companion_support<=1:
        raise ValueError('invalid link thresholds')
    excluded = STOPWORDS | REPORTING_TERMS | PRONOUNS | GENERIC_LINK_WORDS | {t for name in participants for t in terms(name)}
    tokens=[{t for t in terms(text) if t not in excluded and t[1:].isalpha()} for text in texts]
    frequency=Counter(t for ts in tokens for t in ts)
    cores={i:(a,b) for a,b in blocks for i in range(a,b+1)}
    output=[]
    for window in windows:
        hit=window['hit'];local=set(window['fact_numbers']);packets=[]
        a,b=cores[hit]
        for target in range(1,n+1):
            if target in local or cores[target]==cores[hit]:continue
            shared=tokens[hit-1] & tokens[target-1]
            lexical_link=len(shared)>=2 and any(frequency[t]*2<n for t in shared)
            if not lexical_link:continue
            if kernel[hit-1][target-1]<link_floor:continue
            target_a,target_b=cores[target]
            companions=[target]
            for j in range(target_a,target_b+1):
                same_speaker = not speakers or speakers[j-1] == speakers[target-1]
                if same_speaker and support[target-1][j-1]>=companion_support and kernel[target-1][j-1]>=companion_cosine:
                    companions.append(j)
            packets.append({'anchor':target,'shared_terms':sorted(t[1:] for t in shared),
                            'cosine':kernel[hit-1][target-1],'fact_numbers':sorted(set(companions))})
        # A single fact directly across a core boundary may be the missing
        # referent.  This is deliberately one-step only; it cannot pull the
        # whole neighbouring core through a semantic hull.
        edge_candidates = []
        if a > 1: edge_candidates.append(a-1)
        if b < n: edge_candidates.append(b+1)
        for target in edge_candidates:
            if target in local: continue
            shared=tokens[hit-1] & tokens[target-1]
            if shared and kernel[hit-1][target-1] >= bridge_floor:
                local.add(target)
        combined=sorted(local | {i for packet in packets for i in packet['fact_numbers']})
        output.append(window | {'local_fact_numbers':window['fact_numbers'],
                                'supplements':packets,'fact_numbers':combined})
    return output
