"""Resolve relative time expressions in a fact against its source conversation date (program only, no model).

Validated on the LoCoMo holdout as outputs/date-annotation-holdout-20261029 (arm C).

annotate(text, session_date) returns the text with "(<absolute>)" inserted after each expression it can resolve;
the original wording is kept. Vague expressions resolve only as far as they are determined (month, season, year).
"""
import re
from datetime import date, datetime, timedelta

WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
NUM = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
       'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'a couple of': 2, 'couple of': 2, 'a few': 3, 'few': 3, 'several': 3}
SEASONS = {'spring': (3, 5), 'summer': (6, 8), 'fall': (9, 11), 'autumn': (9, 11), 'winter': (12, 2)}


def parse_session(value):
    value = value or ''
    m = re.search(r'(\d{1,2}) ([A-Za-z]+),? (\d{4})', value)
    if m:
        try:
            return datetime.strptime(f'{m[1]} {m[2]} {m[3]}', '%d %B %Y').date()
        except ValueError:
            return None
    m = re.search(r'\b(\d{4})-(\d{2})-(\d{2})', value)
    try:
        return date(int(m[1]), int(m[2]), int(m[3])) if m else None
    except ValueError:
        return None


def date_label(value):
    """Short absolute label for a source date; the raw value when it cannot be parsed."""
    d = parse_session(value)
    return day(d) if d else value


def day(d): return f'{d.day} {d:%B %Y}'
def month(d): return f'{d:%B %Y}'
def add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def season_year(name, d, shift):
    """last: the most recent such season that has ended before the session month; this: the one in the session's
    year; next: the first one starting after the session month. Winter is labelled by the year it starts."""
    start, end = SEASONS[name]
    if name == 'winter':
        this = d.year if d.month == 12 else d.year - 1 if d.month <= 2 else d.year
        last = this - 1 if (d.month == 12 or d.month <= 2) else d.year - 1
        nxt = this + 1 if (d.month == 12 or d.month <= 2) else d.year
        y = {-1: last, 0: this, 1: nxt}[shift]
        return f'winter {y}-{y + 1}'
    y = {-1: d.year if d.month > end else d.year - 1, 0: d.year, 1: d.year if d.month < start else d.year + 1}[shift]
    return f'{name} {y}'


N = r'(\d+|a couple of|couple of|a few|few|several|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)'
UNIT = r'(day|week|month|year)s?'
PATTERNS = [
    (r'\bthe day before yesterday\b', lambda m, d: day(d - timedelta(2))),
    (r'\byesterday\b', lambda m, d: day(d - timedelta(1))),
    (r'\blast night\b', lambda m, d: day(d - timedelta(1))),
    (r'\b(today|tonight|this morning|this afternoon|this evening)\b', lambda m, d: day(d)),
    (r'\btomorrow\b', lambda m, d: day(d + timedelta(1))),
    (r'\blast weekend\b', lambda m, d: 'weekend of ' + day(d - timedelta((d.weekday() - 5) % 7 or 7))),
    (r'\bthis weekend\b', lambda m, d: 'weekend of ' + day(d - timedelta(d.weekday() - 5) if d.weekday() >= 5 else d + timedelta(5 - d.weekday()))),
    (r'\bnext weekend\b', lambda m, d: 'weekend of ' + day(d + timedelta(5 - d.weekday() + (7 if d.weekday() >= 5 else 0)))),
    (r'\blast (' + '|'.join(WEEKDAYS) + r')\b', lambda m, d: day(d - timedelta((d.weekday() - WEEKDAYS.index(m[1].lower())) % 7 or 7))),
    (r'\bnext (' + '|'.join(WEEKDAYS) + r')\b', lambda m, d: day(d + timedelta((WEEKDAYS.index(m[1].lower()) - d.weekday()) % 7 or 7))),
    (r'\blast week\b', lambda m, d: 'week of ' + day(d - timedelta(d.weekday() + 7))),
    (r'\bnext week\b', lambda m, d: 'week of ' + day(d + timedelta(7 - d.weekday()))),
    (r'\bthis week\b', lambda m, d: 'week of ' + day(d - timedelta(d.weekday()))),
    (r'\blast month\b', lambda m, d: month(add_months(d, -1))),
    (r'\bnext month\b', lambda m, d: month(add_months(d, 1))),
    (r'\bthis month\b', lambda m, d: month(d)),
    (r'\blast year\b', lambda m, d: str(d.year - 1)),
    (r'\bnext year\b', lambda m, d: str(d.year + 1)),
    (r'\bthis year\b', lambda m, d: str(d.year)),
    (r'\b(last|this|next) (spring|summer|fall|autumn|winter)\b',
     lambda m, d: season_year(m[2].lower(), d, {'last': -1, 'this': 0, 'next': 1}[m[1].lower()])),
    (r'\b' + N + r' ' + UNIT + r' ago\b', lambda m, d: ago(m, d, -1)),
    (r'\bin ' + N + r' ' + UNIT + r'\b', lambda m, d: ago(m, d, 1)),
]


def ago(m, d, sign):
    n = int(m[1]) if m[1].isdigit() else NUM[m[1].lower()]
    unit = m[2].lower(); approx = m[1].lower() in ('a few', 'few', 'several')
    if unit == 'day': out = day(d + timedelta(sign * n))
    elif unit == 'week': out = day(d + timedelta(sign * 7 * n))
    elif unit == 'month': out = month(add_months(d, sign * n))
    else: out = str(d.year + sign * n)
    return ('around ' if approx or unit in ('week',) else '') + out


COMPILED = [(re.compile(p, re.I), f) for p, f in PATTERNS]


def annotate(text, session_date):
    d = parse_session(session_date)
    if d is None: return text, []
    spans = []
    for rx, fn in COMPILED:
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e, _ in spans): continue
            spans.append((m.start(), m.end(), fn(m, d)))
    out, last = [], 0
    for s, e, value in sorted(spans):
        out += [text[last:e], f' ({value})']; last = e
    out.append(text[last:])
    return ''.join(out), [text[s:e] for s, e, _ in sorted(spans)]

