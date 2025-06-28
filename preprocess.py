import csv, json, lzma, math, os, random
from statistics import NormalDist

INPUT = 'GoodReads_100k_books.csv.xz'
OUTPUT = 'books_trimmed.json'
MAX_ROWS = 15000
random.seed(42)

rows = []
with lzma.open(INPUT, 'rt', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            pages = int(float(r['pages']))
            rating = float(r['rating'])
            reviews = int(float(r['reviews']))
            total = int(float(r['totalratings']))
        except (KeyError, ValueError):
            continue
        if total <= 0:
            continue
        rr = reviews / total if total else 0
        rows.append({'pg': pages, 'rt': rating, 'tr': total, 'rr': rr,
                     't': r.get('title', '').strip(),
                     'au': r.get('author', '').strip()})

# compute bounds
values = {
    'pg': [r['pg'] for r in rows],
    'rt': [r['rt'] for r in rows],
    'tr': [r['tr'] for r in rows],
    'rr': [r['rr'] for r in rows]
}
bounds = {}
for k, vals in values.items():
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    low = vals_sorted[int(n*0.01)]
    high = vals_sorted[int(n*0.99)]
    bounds[k] = (low, high)

filtered = [r for r in rows if
    bounds['pg'][0] <= r['pg'] <= bounds['pg'][1] and
    bounds['rt'][0] <= r['rt'] <= bounds['rt'][1] and
    bounds['tr'][0] <= r['tr'] <= bounds['tr'][1] and
    bounds['rr'][0] <= r['rr'] <= bounds['rr'][1]
]

if len(filtered) > MAX_ROWS:
    filtered = random.sample(filtered, MAX_ROWS)

# compute spearman correlation

def spearman(x, y):
    n = len(x)
    def rank(data):
        order = sorted(range(n), key=lambda i: data[i])
        ranks = [0]*n
        i = 0
        while i < n:
            j = i
            while j+1 < n and data[order[j+1]] == data[order[i]]:
                j += 1
            r = (i + j + 2)/2
            for k in range(i, j+1):
                ranks[order[k]] = r
            i = j+1
        return ranks
    rx = rank(x)
    ry = rank(y)
    mean_rx = sum(rx)/n
    mean_ry = sum(ry)/n
    cov = sum((rx[i]-mean_rx)*(ry[i]-mean_ry) for i in range(n))
    std_rx = math.sqrt(sum((r-mean_rx)**2 for r in rx))
    std_ry = math.sqrt(sum((r-mean_ry)**2 for r in ry))
    r = cov / (std_rx * std_ry)
    if n > 2:
        t = r * math.sqrt((n-2)/(1 - r*r + 1e-9))
        p = 2 * (1 - NormalDist().cdf(abs(t)))
    else:
        p = 1.0
    return r, p

vals_tr = [r['tr'] for r in filtered]
vals_pg = [r['pg'] for r in filtered]
vals_rr = [r['rr'] for r in filtered]
vals_rt = [r['rt'] for r in filtered]

stats_res = {}
for name, arr in [('A', vals_tr), ('B', vals_pg), ('C', vals_rr)]:
    r,p = spearman(arr, vals_rt)
    stats_res[name] = f"\u03c1={r:.2f}, p={p:.2e}"

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, separators=(',', ':'))

size = os.path.getsize(OUTPUT)
print(size)
for k,v in stats_res.items():
    print(f"{k}:{v}")
