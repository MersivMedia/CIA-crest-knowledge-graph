#!/usr/bin/env python3
"""Sample 2 candidates: same strata, docs never sampled before (bench/store/pages_seen.json), more NGA/sci/STARGATE
candidates because those fetch less reliably. Output: the path given as argv[2]"""
import json, sqlite3
import os, sys
# usage: python bench/export_candidates.py <crest.db built from manifest/> <out.json>
HERE = os.path.dirname(os.path.abspath(__file__))
DB, OUT = sys.argv[1], sys.argv[2]
seen = set(json.load(open(os.path.join(HERE, "store", "pages_seen.json"))))
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
strata = {
    "text":     ("collection='General CIA Records' AND content_type IN ('REPORT','MEMO','CABLE','LETTER')", 30, 120),
    "nga":      ("collection='NGA Records (Formerly NIMA)'", 20, 250),
    "photo":    ("collection='Ground Photo Caption Cards'", 20, 120),
    "sci":      ("collection='Scientific Abstracts'", 15, 200),
    "stargate": ("collection='STARGATE'", 15, 200),
}
out = {"want": {}, "docs": [], "excluded_seen": len(seen)}
for s, (where, want, n) in strata.items():
    out["want"][s] = want
    for doc, col, ct, title, pages, pdf in c.execute(
            f"SELECT document_number, collection, content_type, title, pages, COALESCE(pdf_url, file, url) FROM documents "
            f"WHERE {where} AND pages BETWEEN 1 AND 12 ORDER BY RANDOM() LIMIT {n + 50}"):
        if doc in seen or sum(d["stratum"] == s for d in out["docs"]) >= n:
            continue
        out["docs"].append(dict(doc_id=doc, stratum=s, collection=col, content_type=ct, title=title, pages=pages, pdf_url=pdf))
json.dump(out, open(OUT, "w"), indent=1)
from collections import Counter
print(len(out["docs"]), Counter(d["stratum"] for d in out["docs"]), out["want"])
