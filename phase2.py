"""Assignment 2 (Phase 2): Parts A-D in one script.
Usage:
  python phase2.py --data cleaned.csv --col clean_text
Optional: --queries w1,w2,w3  --similar a,b  --unrelated c,d  --analogy a,b,c  --oov misspelledword
Results are printed and also saved in results.txt
"""
import argparse
import os
import math
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--col", default="clean_text")
ap.add_argument("--queries", default="")
ap.add_argument("--similar", default="")
ap.add_argument("--unrelated", default="")
ap.add_argument("--analogy", default="")
ap.add_argument("--oov", default="")
ap.add_argument("--models_dir", default="models")
args = ap.parse_args()

os.makedirs(args.models_dir, exist_ok=True)
_out = open("results.txt", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _out.write(s + "\n")


def sparsity(X):
    return 100.0 * (1 - X.nnz / (X.shape[0] * X.shape[1]))


df = pd.read_csv(args.data).dropna(subset=[args.col])
docs = df[args.col].astype(str).tolist()
tokens = [d.split() for d in docs]
freq = Counter(t for d in tokens for t in d)
log(f"Corpus: {len(docs)} documents, {sum(freq.values())} tokens, {len(freq)} unique words\n")

# ------------------------------------------------------------------ PART A
log("=" * 60, "\nPART A1 - Bag of Words")
cv = CountVectorizer()
Xb = cv.fit_transform(docs)
log("BoW shape (docs x vocab):", Xb.shape)
log(f"BoW sparsity: {sparsity(Xb):.2f}% zeros")

log("\nPART A2 - TF-IDF")
tfv = TfidfVectorizer()
Xt = tfv.fit_transform(docs)
log("TF-IDF shape:", Xt.shape, f"| sparsity {sparsity(Xt):.2f}%")
vocab = np.array(tfv.get_feature_names_out())
idf = tfv.idf_
# pick the first document with at least 8 distinct words as the example
ex = next(i for i in range(len(docs)) if Xt[i].nnz >= 8)
row = Xt[ex].toarray().ravel()
nz = np.nonzero(row)[0]
order = nz[np.argsort(row[nz])]
log(f"Example document #{ex}: {docs[ex][:200]}")
log("HIGH weight words in this doc:", [(vocab[i], round(row[i], 3), "idf=%.2f" % idf[i]) for i in order[::-1][:5]])
log("LOW weight words in this doc :", [(vocab[i], round(row[i], 3), "idf=%.2f" % idf[i]) for i in order[:5]])
low_idf = np.argsort(idf)[:5]
log("Lowest-IDF words in the whole corpus (appear in most docs):",
    [(vocab[i], "idf=%.2f" % idf[i]) for i in low_idf])
log("Reason: IDF = ln((1+N)/(1+df)) + 1 is small when a word occurs in many documents, so its TF-IDF weight shrinks;"
    " rare, document-specific words get large IDF and high weight.")

log("\nPART A3 - N-grams (ngram_range = (1,n))")
rows = []
for n in (1, 2, 3):
    v = CountVectorizer(ngram_range=(1, n))
    X = v.fit_transform(docs)
    rows.append({"n": n, "features": X.shape[1], "sparsity_%": round(sparsity(X), 3)})
tab = pd.DataFrame(rows)
log(tab.to_string(index=False))
tab.to_csv("ngram_table.csv", index=False)
log("Comment: features grow sharply with n and the matrix gets sparser (most bigrams/trigrams occur once).")

log("\nPART A4 - Hand-worked check on a toy sub-sample")
toy = ["the cat sat on the mat", "the dog sat on the log", "the cat chased the dog"]
log("Toy docs:", toy)
tv = TfidfVectorizer()
Xtoy = tv.fit_transform(toy)
fn = list(tv.get_feature_names_out())
N = len(toy)
term, d_idx = "cat", 0
tf = toy[d_idx].split().count(term)
df_t = sum(term in d.split() for d in toy)
idf_t = math.log((1 + N) / (1 + df_t)) + 1
log(f"term='{term}', doc {d_idx}: tf={tf}, df={df_t}, N={N}")
log(f"idf = ln((1+{N})/(1+{df_t})) + 1 = {idf_t:.4f}  -> raw tf-idf = {tf} * {idf_t:.4f} = {tf*idf_t:.4f}")
raw = {}
for w in set(toy[d_idx].split()):
    dfw = sum(w in d.split() for d in toy)
    raw[w] = toy[d_idx].split().count(w) * (math.log((1 + N) / (1 + dfw)) + 1)
norm = math.sqrt(sum(v * v for v in raw.values()))
log(f"L2 norm of doc vector = {norm:.4f}; final weight = {raw[term]:.4f}/{norm:.4f} = {raw[term]/norm:.4f}")
log("Library value:", round(Xtoy[d_idx, fn.index(term)], 4))
bg = CountVectorizer(ngram_range=(2, 2))
Xbg = bg.fit_transform(toy)
bigram = "the cat"
manual_bg = sum(" ".join(toy[i].split()[j:j + 2]) == bigram for i in range(N) for j in range(len(toy[i].split()) - 1))
log(f"Bigram '{bigram}': manual count across toy corpus = {manual_bg}; library count = {Xbg.sum(axis=0).A1[list(bg.get_feature_names_out()).index(bigram)]}")

# ------------------------------------------------------------------ PART B
log("\n" + "=" * 60, "\nPART B - Learned embeddings")
HP = dict(vector_size=100, window=5, min_count=2, epochs=50, seed=42, workers=1)
log("Hyperparameters (same for all):", HP)
cbow = Word2Vec(sentences=tokens, sg=0, **HP)
skip = Word2Vec(sentences=tokens, sg=1, **HP)
ft = FastText(sentences=tokens, sg=1, min_n=3, max_n=6, bucket=50000, **HP)
log("Vocab sizes (min_count=2): CBOW", len(cbow.wv), "| Skip-gram", len(skip.wv), "| fastText", len(ft.wv))

common = [w for w, c in freq.most_common(200) if w in cbow.wv and w in skip.wv and w in ft.wv]
queries = [q for q in args.queries.split(",") if q] or common[:4]

log("\nB4 - OOV / misspelled word demo")
if args.oov:
    oov = args.oov
else:
    base = next(w for w in common if len(w) >= 6)
    oov = base[:2] + base[3:]  # drop one letter -> misspelling
    log(f"(auto-made misspelling of '{base}')")
log(f"Query word: '{oov}'  | in Word2Vec vocab? {oov in cbow.wv} | in fastText vocab? {oov in ft.wv.key_to_index}")
try:
    cbow.wv[oov]
    log("Word2Vec: has a vector")
except KeyError as e:
    log("Word2Vec: KeyError ->", e)
log("fastText neighbours:", [(w, round(s, 3)) for w, s in ft.wv.most_similar(oov, topn=5)])

# ------------------------------------------------------------------ PART C
log("\n" + "=" * 60, "\nPART C1 - Nearest neighbours")
models = {"CBOW": cbow, "Skip-gram": skip, "fastText": ft}
for q in queries:
    for name, m in models.items():
        if q in m.wv.key_to_index:
            log(f"[{name}] {q}: {[(w, round(s, 3)) for w, s in m.wv.most_similar(q, topn=5)]}")

log("\nPART C2 - Cosine similarities")
sim = [w for w in args.similar.split(",") if w] or common[:2]
unr = [w for w in args.unrelated.split(",") if w] or [common[0], common[-1]]
for label, pair in (("expected SIMILAR", sim), ("expected UNRELATED", unr)):
    for name, m in models.items():
        log(f"[{name}] {label} {pair}: {m.wv.similarity(pair[0], pair[1]):.3f}")

log("\nPART C3 - Analogy  a - b + c")
if args.analogy:
    a_, b_, c_ = args.analogy.split(",")
    for name, m in models.items():
        try:
            log(f"[{name}] {a_} - {b_} + {c_} ->", [(w, round(s, 3)) for w, s in m.wv.most_similar(positive=[a_, c_], negative=[b_], topn=3)])
        except KeyError as e:
            log(f"[{name}] failed: {e}")
else:
    log("No --analogy given. Re-run with e.g. --analogy word_a,word_b,word_c (report honestly even if it fails).")

log("\nPART C4 - Count-based vs learned (numbers)")
log(f"TF-IDF dims: {Xt.shape[1]}, sparse ({sparsity(Xt):.1f}% zeros), OOV words ignored.")
log(f"Embedding dims: {HP['vector_size']}, dense, Word2Vec cannot handle OOV; fastText can (sub-words).")

# ------------------------------------------------------------------ PART D
joblib.dump(tfv, os.path.join(args.models_dir, "tfidf.joblib"))
cbow.save(os.path.join(args.models_dir, "cbow.model"))
skip.save(os.path.join(args.models_dir, "skipgram.model"))
ft.save(os.path.join(args.models_dir, "fasttext.model"))
log("\nPART D1 - saved models to:", args.models_dir, os.listdir(args.models_dir))
_out.close()
