"""Part D2 - the interface Phase 3 will call. Loads saved models, never retrains."""
import os
import joblib
import numpy as np
from gensim.models import Word2Vec, FastText
from clean_data import clean_text

MODELS_DIR = "models"
_cache = {}


def _load(method):
    if method not in _cache:
        p = lambda f: os.path.join(MODELS_DIR, f)
        if method == "tfidf":
            _cache[method] = joblib.load(p("tfidf.joblib"))
        elif method == "cbow":
            _cache[method] = Word2Vec.load(p("cbow.model"))
        elif method == "skipgram":
            _cache[method] = Word2Vec.load(p("skipgram.model"))
        elif method == "fasttext":
            _cache[method] = FastText.load(p("fasttext.model"))
        else:
            raise ValueError("method must be one of: tfidf, cbow, skipgram, fasttext")
    return _cache[method]


def vectorize(raw_doc, method="tfidf"):
    """Raw document (string) -> 1-D numpy vector under the chosen representation.
    tfidf: sparse-length vector; cbow/skipgram/fasttext: mean of word vectors."""
    doc = clean_text(raw_doc)
    model = _load(method)
    if method == "tfidf":
        return model.transform([doc]).toarray().ravel()
    wv = model.wv
    toks = [t for t in doc.split() if t in wv.key_to_index] if method != "fasttext" else doc.split()
    if not toks:
        return np.zeros(wv.vector_size)
    return np.mean([wv[t] for t in toks], axis=0)


if __name__ == "__main__":
    for m in ("tfidf", "cbow", "skipgram", "fasttext"):
        v = vectorize("Some raw Document, with Punctuation!!", m)
        print(m, v.shape)
