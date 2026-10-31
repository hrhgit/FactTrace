"""Time ranges for search (program only, no model): what period a question names, and when a fact happened.

Validated offline and by re-answering on the LoCoMo holdout in outputs/mem0-ranking-absorb-20261029 and
outputs/time-lookup-luna-answers-20261029. A fact's times are its source conversation dates, the event dates
relative_dates resolves against them, and dates written in the fact; it matches a question whose range overlaps
any of them. Month names without a year are not parsed.
"""
import calendar
import re
from datetime import date, timedelta

from .relative_dates import SEASONS, WEEKDAYS, annotate, parse_session

MON = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTH = '(' + '|'.join(MON) + ')'
WD = '(' + '|'.join(WEEKDAYS) + ')'
ORD = r'(?:st|nd|rd|th)?'
DAY_A = rf'(\d{{1,2}}){ORD} {MONTH},? (\d{{4}})'  # 3 June, 2023
DAY_B = rf'{MONTH} (\d{{1,2}}){ORD},? (\d{{4}})'  # May 23, 2023
DATE_ANY = rf'(?:{DAY_A}|{DAY_B}|{MONTH},? \d{{4}}|\b(?:19|20)\d\d\b)'
NUMW = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'a': 1}


def month_range(y, m):
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def point(text):
    """One date expression -> (start, end), or None."""
    t = text.lower().strip()
    try:
        if m := re.fullmatch(DAY_A, t):
            d = date(int(m[3]), MON[m[2]], int(m[1])); return d, d
        if m := re.fullmatch(DAY_B, t):
            d = date(int(m[3]), MON[m[1]], int(m[2])); return d, d
    except ValueError:
        return None
    if m := re.fullmatch(rf'{MONTH},? (\d{{4}})', t):
        return month_range(int(m[2]), MON[m[1]])
    if re.fullmatch(r'(19|20)\d\d', t):
        return date(int(t), 1, 1), date(int(t), 12, 31)
    return None


def _mentions(text, taken):
    """Plain date mentions, longest form first, skipping spans already taken."""
    out = []
    for rx in (DAY_A, DAY_B, rf'{MONTH},? \d{{4}}', r'\b(?:19|20)\d\d\b'):
        for m in re.finditer(rx, text, re.I):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            if r := point(m[0]):
                out.append(r); taken.append((m.start(), m.end()))
    return out


def question_ranges(question):
    """Ranges the question names, modifiers first; [] when it names no dated period."""
    low = ' '.join(question.split()).lower()
    out, taken = [], []

    def take(m, r):
        if r and not any(m.start() < e and s < m.end() for s, e in taken):
            out.append(r); taken.append((m.start(), m.end()))

    def within(rx, fn):
        for m in re.finditer(rx, low):
            try:
                take(m, fn(m))
            except ValueError:
                pass

    within(rf'between {MONTH} (\d{{1,2}}){ORD} and {MONTH} (\d{{1,2}}){ORD},? (\d{{4}})',
           lambda m: (date(int(m[5]), MON[m[1]], int(m[2])), date(int(m[5]), MON[m[3]], int(m[4]))))
    within(rf'between {MONTH} and {MONTH},? (\d{{4}})',
           lambda m: (month_range(int(m[3]), MON[m[1]])[0], month_range(int(m[3]), MON[m[2]])[1]))

    def weeks_before(m):
        r, n = point(m[2]), NUMW.get(m[1], 1)
        if not r:
            return None
        return ((r[0] - timedelta(7 * n + 3), r[0] - timedelta(7 * n - 3)) if n > 1
                else (r[0] - timedelta(7), r[0] - timedelta(1)))
    within(rf'(\w+) weeks? before ({DATE_ANY})', weeks_before)

    def weekend(m):
        r = point(m[2])
        if not r:
            return None
        d = r[0]
        sat = (d - timedelta((d.weekday() - 5) % 7 or 7) if m[1] == 'before'
               else d + timedelta((5 - d.weekday()) % 7 or 7))
        return sat, sat + timedelta(1)
    within(rf'weekend (before|after) ({DATE_ANY})', weekend)

    def weekday(m):
        r = point(m[3])
        if not r:
            return None
        d, w = r[0], WEEKDAYS.index(m[1])
        x = (d - timedelta((d.weekday() - w) % 7 or 7) if m[2] == 'before'
             else d + timedelta((w - d.weekday()) % 7 or 7))
        return x, x
    within(rf'the {WD} (before|after) ({DATE_ANY})', weekday)

    def part_of_month(m):
        a, b = month_range(int(m[4]), MON[m[3]]); k = m[1]
        if k == 'last week': return b - timedelta(6), b
        if k == 'last two weeks': return b - timedelta(13), b
        if k in ('first half', 'beginning'): return a, a + timedelta(14)
        if k in ('second half', 'end'): return a + timedelta(15), b
        if k == 'first weekend':
            sat = a + timedelta((5 - a.weekday()) % 7); return sat, sat + timedelta(1)
        x = a + timedelta((WEEKDAYS.index(m[2]) - a.weekday()) % 7); return x, x
    within(rf'(last week|last two weeks|first weekend|first half|second half|beginning|end|first {WD}) of {MONTH},? (\d{{4}})',
           part_of_month)
    out += _mentions(' '.join(question.split()), taken)
    return out


def _label_range(label):
    """A relative_dates label ("week of 8 May 2023", "around 3 June 2022", "summer 2021") -> range."""
    s = label.lower(); approx = s.startswith('around ')
    s = s[7:] if approx else s
    if (m := re.fullmatch(r'weekend of (.+)', s)) and (r := point(m[1])):
        return r[0], r[0] + timedelta(1)
    if (m := re.fullmatch(r'week of (.+)', s)) and (r := point(m[1])):
        return r[0], r[0] + timedelta(6)
    if m := re.fullmatch(r'(spring|summer|fall|autumn) (\d{4})', s):
        a, b = SEASONS[m[1]]; return month_range(int(m[2]), a)[0], month_range(int(m[2]), b)[1]
    if m := re.fullmatch(r'winter (\d{4})-(\d{4})', s):
        return date(int(m[1]), 12, 1), month_range(int(m[2]), 2)[1]
    r = point(s)
    return (r[0] - timedelta(3), r[1] + timedelta(3)) if r and approx else r


def fact_ranges(text, session_dates):
    """Source conversation days, resolved relative expressions and written dates of one fact."""
    out = []
    for value in session_dates:
        d = parse_session(value)
        if d is None:
            continue
        out.append((d, d))
        annotated, _ = annotate(text, value)
        out += [r for label in re.findall(r'\(([^()]+)\)', annotated) if (r := _label_range(label))]
    return out + _mentions(text, [])


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]
