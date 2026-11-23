"""TF-IDF retrieval baseline (proposal Sec. IV.D, Config A) and Config B candidate feed.

Each product is encoded as one text document built from its name, brand,
category (display name, leaf id and group), ingredient text and label text
(dietary tags + raw Open Food Facts labels). Products are ranked by cosine
similarity between the query vector and the product vector, Eq. (4):

    sim(q, d) = (v_q . v_d) / (||v_q|| ||v_d||)

scikit-learn's TfidfVectorizer L2-normalises every row, so the dot product
(`linear_kernel`) equals the cosine similarity.

Design rules (keep the baseline a *pure lexical* retriever):
  * price, allergens and barcodes are NOT indexed (the proposal's fields are
    name, category, ingredient text and label text);
  * no hidden constraint filtering; Config A always returns its top product,
    so its Infeasible-Request Accuracy is 0 by construction;
  * ties are broken by barcode, exactly like the KG retrieval query.

The vectorizer settings in DEFAULT_VECTORIZER were frozen on the 20 development
prompts (data/evaluation/constraint_prompts.json) BEFORE the 80 test prompts
were written; see scripts/tune_tfidf_dev.py for the comparison.
"""
from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from clickless_kg import vocab

#: Frozen on the development prompts only (see module docstring). These are
#: scikit-learn's defaults, written out so the paper can state them: unigrams,
#: raw term counts, smoothed idf = ln((1+n)/(1+df)) + 1, L2-normalised rows.
#: On the 18 feasible dev prompts no variant (sublinear tf, accent stripping,
#: English stop words, bigrams) changed top-1 accuracy by more than one prompt,
#: and all reach Recall@20 = 18/18 (scripts/tune_tfidf_dev.py), so the library
#: defaults are kept rather than fitting 18 prompts.
DEFAULT_VECTORIZER = {
    "lowercase": True,
    "strip_accents": None,
    "ngram_range": (1, 1),
    "sublinear_tf": False,
    "smooth_idf": True,
    "norm": "l2",
}

#: Number of candidates handed to the Config B LLM (proposal Sec. V.B).
CONFIG_B_DEPTH = 20


def _label_text(tag: str) -> str:
    """'en:no-gmos' -> 'no gmos'; 'fr:sans-gluten' -> 'sans gluten'."""
    return re.sub(r"[-_:]+", " ", tag.split(":", 1)[-1]).strip()


def build_document(product: dict) -> str:
    """The text that represents one catalog product in the TF-IDF index."""
    leaf = product["category"]
    spec = vocab.CATEGORY_BY_NAME.get(leaf)
    category_text = [leaf.replace("_", " ")]
    if spec is not None:
        category_text += [spec.display, spec.group]
    parts: list[str] = [
        product.get("name") or "",
        product.get("brand") or "",
        " ".join(category_text),
        product.get("ingredients_text") or "",
        " ".join(t.replace("_", " ") for t in product.get("dietary_tags", [])),
        " ".join(_label_text(t) for t in product.get("labels_raw", [])),
    ]
    return " \n".join(p for p in parts if p)


class TfidfIndex:
    """TF-IDF index over the catalog; `search` returns ranked catalog dicts."""

    def __init__(self, catalog: list[dict], **vectorizer_kwargs):
        if not catalog:
            raise ValueError("catalog is empty")
        settings = {**DEFAULT_VECTORIZER, **vectorizer_kwargs}
        self.catalog = list(catalog)
        self.vectorizer = TfidfVectorizer(**settings)
        self.matrix = self.vectorizer.fit_transform(build_document(p) for p in self.catalog)

    def scores(self, query: str):
        """Cosine similarity of `query` to every product (catalog order)."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        return linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()

    def search(self, query: str, k: int | None = 10) -> list[tuple[dict, float]]:
        """Top-k (product, score) pairs, highest score first, ties by barcode."""
        sims = self.scores(query)
        order = sorted(range(len(self.catalog)),
                       key=lambda i: (-sims[i], self.catalog[i]["code"]))
        if k is not None:
            order = order[:k]
        return [(self.catalog[i], float(sims[i])) for i in order]

    def rank_of_first(self, query: str, acceptable: Iterable[str]) -> int | None:
        """1-based rank of the first acceptable product in the full ranking (None if absent)."""
        wanted = set(acceptable)
        if not wanted:
            return None
        for rank, (p, _) in enumerate(self.search(query, k=None), start=1):
            if p["code"] in wanted:
                return rank
        return None


def config_a(query: str, index: TfidfIndex) -> dict:
    """Config A: the top-ranked TF-IDF product (never 'no match')."""
    return index.search(query, k=1)[0][0]


def tfidf_candidates(query: str, index: TfidfIndex, k: int = CONFIG_B_DEPTH) -> list[dict]:
    """Ranked catalog dicts for Config B (`clickless_llm.config_b.select_from_candidates`).

    The dicts are the catalog objects themselves (not copies), so Config B's
    identity-preserving return value can be matched against the catalog.
    """
    return [p for p, _ in index.search(query, k=k)]
