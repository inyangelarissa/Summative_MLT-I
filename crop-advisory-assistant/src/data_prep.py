#!/usr/bin/env python
"""Stage 1: raw documents -> cleaned, chunked, tagged passages (data/interim/chunks.jsonl).

Put your source documents in data/raw/ (.pdf .txt .md .html) and describe them in
data/raw/sources.csv (columns: filename,title,publisher,url,license,crop,year). The
manifest is how you document provenance and licences for the report.

  python src/data_prep.py --raw_dir data/raw --out data/interim/chunks.jsonl
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge import detect  # noqa: E402


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self.skip += 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext in (".html", ".htm"):
        ex = _TextExtractor()
        ex.feed(path.read_text(encoding="utf-8", errors="ignore"))
        return "".join(ex.parts)
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit("pip install pypdf   (needed to read PDFs)")
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"unsupported file type: {path}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                      # de-hyphenate line breaks
    # cut the reference list if it appears in the second half of the document
    m = None
    for m in re.finditer(r"^\s*(references|bibliography|literature cited)\s*$", text, flags=re.I | re.M):
        pass
    if m and m.start() > 0.5 * len(text):
        text = text[: m.start()]
    lines = text.split("\n")
    freq = Counter(l.strip() for l in lines if 0 < len(l.strip()) < 80)
    kept = []
    for l in lines:
        s = l.strip()
        if re.fullmatch(r"(page\s*)?\d{1,4}", s, flags=re.I):          # page numbers
            continue
        if s and freq[s] >= 4:                                         # repeated headers/footers
            continue
        kept.append(s)
    text = "\n".join(kept)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\n{2,}", "<<P>>", text)                            # paragraph breaks
    text = re.sub(r"\n", " ", text)                                    # soft wraps
    text = text.replace("<<P>>", "\n\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(paragraph: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", paragraph) if s.strip()]


def chunk_text(text, target_words=150, max_words=220, overlap_sentences=1, min_words=40):
    sents = []
    for para in text.split("\n\n"):
        for s in split_sentences(para):
            w = s.split()
            while len(w) > max_words:                                  # very long "sentence"
                sents.append(" ".join(w[:max_words]))
                w = w[max_words:]
            if w:
                sents.append(" ".join(w))
    chunks, cur, n = [], [], 0
    for s in sents:
        cur.append(s)
        n += len(s.split())
        if n >= target_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            n = sum(len(x.split()) for x in cur)
    if cur and n >= min_words and (not chunks or " ".join(cur) not in chunks[-1]):
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= min_words]


def load_manifest(raw_dir: Path):
    path = raw_dir / "sources.csv"
    if not path.exists():
        print(f"WARNING: {path} not found. Provenance/licence columns will be empty; fill it in for your report.")
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out", default="data/interim/chunks.jsonl")
    ap.add_argument("--target_words", type=int, default=150)
    ap.add_argument("--max_words", type=int, default=220)
    ap.add_argument("--min_words", type=int, default=40)
    ap.add_argument("--keep_general", action="store_true",
                    help="Keep chunks that mention no known crop or disease (default: drop them)")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    manifest = load_manifest(raw)
    files = sorted(p for p in raw.iterdir() if p.suffix.lower() in (".pdf", ".txt", ".md", ".html", ".htm"))
    if not files:
        raise SystemExit(f"No documents found in {raw}")

    seen, rows, dropped_dup, dropped_general = set(), [], 0, 0
    for path in files:
        meta = manifest.get(path.name, {})
        if manifest and not meta:
            print(f"WARNING: {path.name} is not listed in sources.csv")
        text = clean_text(read_text(path))
        for i, chunk in enumerate(chunk_text(text, args.target_words, args.max_words, min_words=args.min_words)):
            key = hashlib.md5(re.sub(r"\W+", " ", chunk.lower()).encode()).hexdigest()
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            crop, pairs = detect(chunk, meta.get("crop"))
            if crop == "general" and not args.keep_general:
                dropped_general += 1
                continue
            rows.append({
                "chunk_id": f"{path.stem}::{i:04d}", "source_doc": path.stem, "crop": crop,
                "diseases": sorted({d for _, d in pairs}), "disease_pairs": [list(p) for p in pairs],
                "text": chunk, "n_words": len(chunk.split()), "title": meta.get("title", ""),
                "license": meta.get("license", ""), "url": meta.get("url", ""),
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(files)} documents -> {len(rows)} chunks (dropped: {dropped_dup} duplicates, "
          f"{dropped_general} with no crop/disease mention)")
    print("chunks per crop:", dict(Counter(r["crop"] for r in rows)))
    print("chunks mentioning a disease:", sum(1 for r in rows if r["diseases"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
