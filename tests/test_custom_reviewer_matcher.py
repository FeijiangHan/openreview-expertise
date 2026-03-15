import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "expertise" / "custom_reviewer_matcher.py"
spec = importlib.util.spec_from_file_location("custom_reviewer_matcher", MODULE_PATH)
custom_reviewer_matcher = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(custom_reviewer_matcher)

ReviewerRecord = custom_reviewer_matcher.ReviewerRecord
PaperRecord = custom_reviewer_matcher.PaperRecord
_normalize_reference_title = custom_reviewer_matcher._normalize_reference_title
build_reviewer_embeddings = custom_reviewer_matcher.build_reviewer_embeddings
build_reviewer_pool_from_references = custom_reviewer_matcher.build_reviewer_pool_from_references
parse_title_abstract_references = custom_reviewer_matcher.parse_title_abstract_references
rank_reviewers = custom_reviewer_matcher.rank_reviewers


def test_parse_title_abstract_and_references():
    text = """My Great Paper

Abstract
We propose a method.
It works well.

1 Introduction
Something.

References
[1] A. Author. Foundation Models for Science. Journal.
[2] B. Author. Better Benchmarks. Conf.
"""
    title, abstract, refs = parse_title_abstract_references(text)
    assert title == "My Great Paper"
    assert "We propose a method." in abstract
    assert len(refs) == 2


def test_normalize_reference_title():
    ref = "[12] Smith, J. Large Language Models for Peer Review. NeurIPS."
    normalized = _normalize_reference_title(ref)
    assert normalized == "Large Language Models for Peer Review"


def test_build_reviewer_pool_deduplicates(monkeypatch):
    def fake_fetch(title_query, timeout_s=15):
        return {
            "paperId": "p1",
            "title": "Paper 1",
            "abstract": "Abstract 1",
            "authors": [
                {"authorId": "a1", "name": "Alice"},
                {"authorId": "a1", "name": "Alice"},
            ],
        }

    monkeypatch.setattr(custom_reviewer_matcher, "fetch_semantic_scholar_paper", fake_fetch)

    pool = build_reviewer_pool_from_references(
        [
            "[1] X. Y. Paper 1. Venue.",
            "[2] X. Y. Paper 1. Venue.",
        ]
    )

    assert len(pool) == 1
    assert pool[0].reviewer_id == "a1"
    assert pool[0].paper_count == 1


class FakeEmbedder:
    def encode_texts(self, texts, batch_size=8):
        vectors = []
        for text in texts:
            if "topic-a" in text:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return np.array(vectors, dtype=np.float32)


def test_rank_reviewers_orders_by_similarity():
    reviewers = [
        ReviewerRecord("r1", "Alice", "", "", 1, [PaperRecord("p1", "topic-a", "good")]),
        ReviewerRecord("r2", "Bob", "", "", 1, [PaperRecord("p2", "topic-b", "other")]),
    ]
    embedder = FakeEmbedder()
    emb = build_reviewer_embeddings(reviewers, embedder)
    ranked = rank_reviewers("topic-a", "new", reviewers, emb, embedder, top_n=2)

    assert ranked[0]["reviewer_id"] == "r1"
    assert ranked[0]["score"] > ranked[1]["score"]
