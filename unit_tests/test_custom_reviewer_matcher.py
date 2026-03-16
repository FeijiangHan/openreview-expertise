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
enrich_reviewer_profiles = custom_reviewer_matcher.enrich_reviewer_profiles
extract_pdf_metadata = custom_reviewer_matcher.extract_pdf_metadata
_grobid_tei_to_fields = custom_reviewer_matcher._grobid_tei_to_fields
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


def test_parse_grobid_tei_fields():
    tei = """<TEI xmlns=\"http://www.tei-c.org/ns/1.0\">
      <teiHeader>
        <fileDesc><titleStmt><title>Grobid Title</title></titleStmt></fileDesc>
        <profileDesc><abstract><p>Abstract line one.</p><p>Line two.</p></abstract></profileDesc>
      </teiHeader>
      <text><back><listBibl>
        <biblStruct><analytic><title>Ref A</title></analytic></biblStruct>
        <biblStruct><monogr><title>Ref B</title></monogr></biblStruct>
      </listBibl></back></text>
    </TEI>"""
    title, abstract, refs = _grobid_tei_to_fields(tei)
    assert title == "Grobid Title"
    assert "Abstract line one." in abstract
    assert refs == ["Ref A", "Ref B"]


def test_extract_pdf_metadata_auto_falls_back(monkeypatch):
    monkeypatch.setattr(custom_reviewer_matcher, "parse_pdf_with_grobid", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(custom_reviewer_matcher, "extract_text_from_pdf", lambda *_args, **_kwargs: "T\n\nAbstract\nA\n\nReferences\n[1] X. Y. Z.")
    title, abstract, refs = extract_pdf_metadata(Path("dummy.pdf"), parser="auto")
    assert title == "T"
    assert abstract == "A"
    assert len(refs) == 1


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

    pool = build_reviewer_pool_from_references([
        "[1] X. Y. Paper 1. Venue.",
        "[2] X. Y. Paper 1. Venue.",
    ])

    assert len(pool) == 1
    assert pool[0].reviewer_id == "a1"
    assert pool[0].paper_count == 1


def test_enrich_reviewer_profiles_openalex(monkeypatch):
    reviewers = [ReviewerRecord("name::alice", "Alice", "", "", 1, [])]
    monkeypatch.setattr(
        custom_reviewer_matcher,
        "fetch_openalex_author_profile",
        lambda *_args, **_kwargs: {"affiliation": "MIT", "homepage": "https://mit.edu"},
    )
    result = enrich_reviewer_profiles(reviewers, source="openalex")
    assert result[0].affiliation == "MIT"
    assert result[0].homepage == "https://mit.edu"


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


def test_package_import_allows_custom_matcher_module():
    import importlib
    import sys

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    module = importlib.import_module("expertise.custom_reviewer_matcher")
    assert hasattr(module, "run_pdf_matching")
