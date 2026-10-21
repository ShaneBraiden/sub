"""Train the sentence embeddings of the extractive summarizer (src/subfix/summarize.py, method `embed`).

LSA (Deerwester et al. 1990): TF-IDF of content-word stems (sublinear tf, smoothed idf, terms in at
least `min_df` documents) over a background corpus, and its truncated SVD; a sentence embedding is its
TF-IDF vector projected on the top `dims` right singular vectors. The corpus mixes written and spoken
English: the paragraphs of the Brown corpus (500 texts, 15 genres; already used for our POS tagger)
and windows of ~10 sentences of the AMI training meetings (95 meetings; the dev and test meetings of
the summary evaluation are not used).

  python experiments/train_summarizer.py [--dims 200]
Writes models/summarizer/lsa.npz (terms, idf, components) + models/summarizer/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from subfix.subio import Word  # noqa: E402
from subfix.summarize import content_terms, make_sentences  # noqa: E402

OUT = ROOT / "models" / "summarizer"
AMI = ROOT / "data" / "ami"
WINDOW = 10            # AMI sentences per LSA document


def ami_words(rec: dict) -> list[Word]:
    return [Word(t, s, e, meta={"speaker": spk}) for t, s, e, spk in rec["words"]]


def ami_meetings(splits: tuple[str, ...]) -> list[dict]:
    man = json.loads((AMI / "manifest.json").read_text(encoding="utf-8"))
    return [json.loads((AMI / "meetings" / f"{m['id']}.json").read_text(encoding="utf-8"))
            for m in man if m["split"] in splits]


def corpus(min_terms: int = 5) -> tuple[list[list[str]], dict]:
    """Background documents as content-term lists: Brown paragraphs + AMI training-meeting windows."""
    from nltk.corpus import brown
    docs, n_brown, n_ami = [], 0, 0
    for para in brown.paras():
        terms = content_terms([w for sent in para for w in sent])
        if len(terms) >= min_terms:
            docs.append(terms)
            n_brown += 1
    for rec in ami_meetings(("train",)):
        sents = make_sentences(ami_words(rec))
        for k in range(0, len(sents), WINDOW):
            terms = [t for s in sents[k:k + WINDOW] for t in s.terms]
            if len(terms) >= min_terms:
                docs.append(terms)
                n_ami += 1
    return docs, {"brown_paragraphs": n_brown, "ami_windows": n_ami}


def fit(docs: list[list[str]], dims: int, min_df: int = 3, seed: int = 0) -> dict:
    """TF-IDF + truncated SVD of the documents -> {terms, idf, components, explained}."""
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    terms = sorted(t for t, c in df.items() if c >= min_df)
    index = {t: i for i, t in enumerate(terms)}
    n = len(docs)
    idf = np.asarray([math.log((1 + n) / (1 + df[t])) + 1.0 for t in terms], dtype=np.float32)
    rows, cols, vals = [], [], []
    for r, d in enumerate(docs):
        counts: dict[int, int] = {}
        for t in d:
            j = index.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            rows.append(r)
            cols.append(j)
            vals.append((1 + math.log(c)) * idf[j])
    X = csr_matrix((vals, (rows, cols)), shape=(n, len(terms)), dtype=np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = csr_matrix(X.multiply(1.0 / np.maximum(norms, 1e-9)[:, None]))
    svd = TruncatedSVD(n_components=dims, algorithm="randomized", n_iter=7, random_state=seed)
    svd.fit(X)
    return {"terms": terms, "idf": idf, "components": svd.components_.astype(np.float32),
            "explained": float(svd.explained_variance_ratio_.sum())}


def save(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, terms=np.asarray(model["terms"]), idf=model["idf"], components=model["components"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dims", type=int, default=200)
    ap.add_argument("--min-df", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    docs, info = corpus()
    model = fit(docs, args.dims, args.min_df)
    save(model, OUT / "lsa.npz")
    metrics = {"dims": args.dims, "min_df": args.min_df, "documents": len(docs), **info,
               "vocabulary": len(model["terms"]), "explained_variance": round(model["explained"], 4),
               "seconds": round(time.time() - t0, 1)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
