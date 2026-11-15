# NLP Assignment 2 - Word Representations

## Run steps
```
pip install -r requirements.txt
# (only if you have no cleaned dataset yet)
python clean_data.py --raw raw.csv --col text --out cleaned.csv
python phase2.py --data cleaned.csv --col clean_text \
    --queries word1,word2,word3 --similar wordA,wordB --unrelated wordC,wordD \
    --analogy a,b,c --oov misspelledword
python vectorizer.py      # quick test of the vectorize(doc, method) interface
```
Outputs: `results.txt`, `ngram_table.csv`, `models/` (tfidf.joblib, cbow.model, skipgram.model, fasttext.model).

## Interface for Phase 3
```python
from vectorizer import vectorize
vec = vectorize("raw document text", method="tfidf")  # tfidf | cbow | skipgram | fasttext
```
Git tag: `git tag assignment-2 && git push --tags`
