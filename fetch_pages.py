#!/usr/bin/env python3
"""
Fetch benchmark pages for embed_bench.py: page images + reference transcriptions.

Source order per document (same as the pipeline, see PRD §3):
  1. archive.org  - PDF + ABBYY djvu XML (text per page)
  2. Wayback      - the cia.gov PDF via web.archive.org/web/<ts>id_/<url>; image-only, no text layer
Keeps the first `want[stratum]` documents that resolve, renders up to --max-pages pages at 150 DPI.
Bytes come from archive.org/Wayback, but every page record cites the OFFICIAL source: the cia.gov reading-room
PDF from the manifest, with a 1-based #page= anchor (cia_pdf_url / citation). archive.org URLs are kept only as
`fetched_from` provenance and are never shown as the citation.

    python fetch_pages.py --candidates bench_candidates.json --out-dir /root/pages --workers 6
Writes <out-dir>/pages.json: [{doc_id, page, stratum, collection, title, abbyy, fetched_from, image,
                                cia_doc_url, cia_pdf_url, citation}]   (page is 0-based; citation anchor is 1-based)
"""
import argparse, json, os, subprocess, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
UA = "Mozilla/5.0"

def curl(url, out=None, timeout=120):
    cmd = ["curl", "-sL", "--compressed", "--max-time", str(timeout), "-A", UA, "-o", out or "-", url]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout + 30).stdout
    except Exception:
        return b""

def cia_urls(doc, manifest_url):
    """Official citation targets. Manifest URLs use the 2017 /library/readingroom/ form; the live site drops /library."""
    pdf = (manifest_url or f"https://www.cia.gov/readingroom/docs/{doc.lower()}.pdf").replace("/library/readingroom/", "/readingroom/")
    if pdf.startswith("http://"):
        pdf = "https://" + pdf[len("http://"):]
    return pdf, f"https://www.cia.gov/readingroom/document/{doc.lower()}"


def is_pdf(path):
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False

def from_archive(doc, wd):
    try:
        m = json.loads(curl(f"https://archive.org/metadata/{doc}", timeout=40) or b"{}")
    except Exception:
        return None
    files = {f.get("format"): f["name"] for f in m.get("files", [])}
    pdf_n = files.get("Image Container PDF") or files.get("Text PDF")
    if not pdf_n:
        return None
    base = f"https://{m.get('server')}{m.get('dir')}/"
    pdf = os.path.join(wd, f"{doc}.pdf")
    curl(base + urllib.parse.quote(pdf_n), out=pdf, timeout=300)
    if not is_pdf(pdf):
        return None
    texts = []
    if files.get("Djvu XML"):
        raw = curl(base + urllib.parse.quote(files["Djvu XML"]), timeout=300).decode("utf-8", "replace")
        if raw.lstrip().startswith("<"):
            try:
                texts = [" ".join(w.text for w in o.iter("WORD") if w.text) for o in ET.fromstring(raw).iter("OBJECT")]
            except ET.ParseError:
                pass
    return pdf, texts, "archive.org"

def from_wayback(doc, url, wd):
    """cia.gov PDF via Wayback. Snapshots are keyed by the exact archived path (usually UPPERCASE doc id), and the
    availability API is unreliable with encoded URLs, so request id_ captures directly: web.archive.org/web/2id_/<url>
    redirects to the nearest capture. Tries the uppercase and lowercase paths."""
    if not url:
        return None
    url = url.replace("/library/readingroom/", "/readingroom/").replace("http://", "https://")
    base = url.rsplit("/", 1)[0]
    pdf = os.path.join(wd, f"{doc}.pdf")
    for u in (f"{base}/{doc.upper()}.pdf", f"{base}/{doc.lower()}.pdf"):
        curl(f"https://web.archive.org/web/2id_/{u}", out=pdf, timeout=300)
        if is_pdf(pdf):
            return pdf, [], "wayback"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pages", type=int, default=3); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import pymupdf
    os.makedirs(a.out_dir, exist_ok=True)
    C = json.load(open(a.candidates)); want = C["want"]
    got = {s: 0 for s in want}; lock = threading.Lock(); out = []; stats = {"archive.org": 0, "wayback": 0, "none": 0}

    def work(d):
        s = d["stratum"]
        with lock:
            if got[s] >= want[s]:
                return
        r = from_archive(d["doc_id"], a.out_dir) or from_wayback(d["doc_id"], d.get("pdf_url"), a.out_dir)
        with lock:
            if not r:
                stats["none"] += 1; return
            if got[s] >= want[s]:
                return
            got[s] += 1; stats[r[2]] += 1
        pdf, texts, src = r
        try:
            pd = pymupdf.open(pdf)
        except Exception:
            return
        rows = []
        cia_pdf, cia_doc = cia_urls(d["doc_id"], d.get("pdf_url"))
        for i in range(min(a.max_pages, pd.page_count)):
            img = os.path.join(a.out_dir, f"{d['doc_id']}_p{i}.jpg")
            pd[i].get_pixmap(dpi=150).save(img)
            rows.append(dict(doc_id=d["doc_id"], page=i, stratum=s, collection=d["collection"], title=d["title"],
                             abbyy=texts[i] if i < len(texts) else "", fetched_from=src, image=img,
                             cia_doc_url=cia_doc, cia_pdf_url=cia_pdf, citation=f"{cia_pdf}#page={i + 1}"))
        with lock:
            out.extend(rows)
        print(f"{d['doc_id']} {s:<8} {src:<11} pages={len(rows)} text={'y' if texts else 'n'}", flush=True)

    t = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, C["docs"]))
    json.dump(out, open(os.path.join(a.out_dir, "pages.json"), "w"), indent=1)
    print(f"{len(out)} pages from {sum(got.values())} docs in {time.time() - t:.0f}s; per stratum {got}; sources {stats}")
    print(sum(1 for p in out if len(p["abbyy"]) > 200), "pages with usable ABBYY text (comparison only)")

if __name__ == "__main__":
    main()
