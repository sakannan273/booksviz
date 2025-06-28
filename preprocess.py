import csv
import json
import os
import random

INPUT = 'GoodReads_100k_books.csv'
OUTPUT = 'scatter_data.json'

rows = []
with open(INPUT, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            pages = int(float(r['pages']))
            reviews = int(float(r['reviews']))
            rating = float(r['rating'])
        except (ValueError, KeyError):
            continue
        desc = r.get('desc') or ''
        blurb = len(desc)
        rows.append({'pages': pages, 'blurb': blurb, 'reviews': reviews, 'rating': rating})

# compute percentiles
cols = {
    'pages': [r['pages'] for r in rows],
    'blurb': [r['blurb'] for r in rows],
    'reviews': [r['reviews'] for r in rows],
    'rating': [r['rating'] for r in rows]
}

bounds = {}
for k, data in cols.items():
    data_sorted = sorted(data)
    n = len(data_sorted)
    lower_idx = int(n * 0.005)
    upper_idx = int(n * 0.995)
    bounds[k] = (data_sorted[lower_idx], data_sorted[upper_idx])

filtered = [r for r in rows if
    bounds['pages'][0] <= r['pages'] <= bounds['pages'][1] and
    bounds['blurb'][0] <= r['blurb'] <= bounds['blurb'][1] and
    bounds['reviews'][0] <= r['reviews'] <= bounds['reviews'][1] and
    bounds['rating'][0] <= r['rating'] <= bounds['rating'][1]]

random.seed(42)
random.shuffle(filtered)
filtered = filtered[:5000]

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, separators=(',', ':'))

size = os.path.getsize(OUTPUT)
print(size)
