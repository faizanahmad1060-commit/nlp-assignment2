"""Minimal Phase-1 style cleaning. Use ONLY if you do not already have a cleaned dataset.
Usage: python clean_data.py --raw raw.csv --col text --out cleaned.csv
"""
import argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text, remove_stopwords=True):
    """lowercase -> remove urls -> keep letters only -> drop short tokens (+ stopwords)."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--col", default="text")
    ap.add_argument("--out", default="cleaned.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.raw)
    df = df.dropna(subset=[a.col]).drop_duplicates(subset=[a.col])
    df["clean_text"] = df[a.col].map(clean_text)
    df = df[df["clean_text"].str.split().str.len() > 0]
    df[["clean_text"]].to_csv(a.out, index=False)
    print(f"Saved {len(df)} cleaned documents to {a.out}")
