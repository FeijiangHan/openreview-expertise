import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    abstract: str


@dataclass
class ReviewerRecord:
    reviewer_id: str
    name: str
    affiliation: str
    homepage: str
    paper_count: int
    papers: List[PaperRecord]


class PDFExtractionError(RuntimeError):
    pass


def _request_json(
    url: str,
    timeout_s: int = 15,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
    retries: int = 2,
    retry_sleep_s: float = 0.5,
) -> Optional[Dict]:
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"User-Agent": "openreview-expertise-custom-matcher/1.1", **(headers or {})},
    )

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, urllib.error.HTTPError):
            if attempt >= retries:
                return None
            time.sleep(retry_sleep_s * (attempt + 1))
    return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF when available."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise PDFExtractionError(
            "PyMuPDF (fitz) is required for PDF parsing. Install with `pip install pymupdf`."
        ) from exc

    text_parts: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))

    text = "\n".join(text_parts).strip()
    if not text:
        raise PDFExtractionError(f"No text extracted from PDF: {pdf_path}")
    return text


def _build_multipart_pdf_request(pdf_path: Path, boundary: str = "----openreviewmatcher") -> Tuple[bytes, str]:
    file_bytes = pdf_path.read_bytes()
    header = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"input\"; filename=\"{pdf_path.name}\"\r\n"
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + file_bytes + footer
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _grobid_tei_to_fields(tei_xml: str) -> Tuple[str, str, List[str]]:
    root = ET.fromstring(tei_xml)
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    title = ""
    title_node = root.find(".//tei:titleStmt/tei:title", ns)
    if title_node is not None and title_node.text:
        title = title_node.text.strip()

    abstract_parts = []
    abstract_node = root.find(".//tei:profileDesc/tei:abstract", ns)
    if abstract_node is not None:
        for p in abstract_node.findall(".//tei:p", ns):
            if p.text and p.text.strip():
                abstract_parts.append(p.text.strip())
    abstract = " ".join(abstract_parts).strip()

    references: List[str] = []
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", ns):
        ref_title = ""
        title_in_ref = bibl.find(".//tei:analytic/tei:title", ns)
        if title_in_ref is None:
            title_in_ref = bibl.find(".//tei:monogr/tei:title", ns)
        if title_in_ref is not None and title_in_ref.text:
            ref_title = re.sub(r"\s+", " ", title_in_ref.text).strip()
        if ref_title:
            references.append(ref_title)

    return title, abstract, references


def parse_pdf_with_grobid(pdf_path: Path, grobid_url: str, timeout_s: int = 60) -> Optional[Tuple[str, str, List[str]]]:
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    body, content_type = _build_multipart_pdf_request(pdf_path)
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "User-Agent": "openreview-expertise-custom-matcher/1.1",
            "Content-Type": content_type,
            "Accept": "application/xml",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            xml_payload = resp.read().decode("utf-8", errors="ignore")
        parsed = _grobid_tei_to_fields(xml_payload)
        if parsed[0] or parsed[1] or parsed[2]:
            return parsed
    except (urllib.error.URLError, TimeoutError, ET.ParseError):
        return None
    return None


def parse_title_abstract_references(pdf_text: str) -> Tuple[str, str, List[str]]:
    """Heuristic extraction of title, abstract and reference strings from raw PDF text."""
    lines = [line.strip() for line in pdf_text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    if not non_empty_lines:
        raise ValueError("PDF text is empty after normalization")

    title = non_empty_lines[0]

    abstract = ""
    abstract_header_idx = None
    for idx, line in enumerate(lines[:250]):
        if re.fullmatch(r"abstract", line, flags=re.IGNORECASE):
            abstract_header_idx = idx
            break

    if abstract_header_idx is not None:
        abstract_lines = []
        for line in lines[abstract_header_idx + 1 :]:
            if re.fullmatch(r"(1\.?\s+)?introduction", line, flags=re.IGNORECASE):
                break
            if re.fullmatch(r"keywords", line, flags=re.IGNORECASE):
                break
            if re.fullmatch(r"references|bibliography", line, flags=re.IGNORECASE):
                break
            if line:
                abstract_lines.append(line)
        abstract = " ".join(abstract_lines).strip()

    references = _extract_reference_lines(lines)
    return title, abstract, references


def _extract_reference_lines(lines: Sequence[str]) -> List[str]:
    ref_start = None
    for idx, line in enumerate(lines):
        if re.fullmatch(r"references|bibliography", line, flags=re.IGNORECASE):
            ref_start = idx + 1
            break

    if ref_start is None:
        return []

    ref_lines: List[str] = []
    for raw_line in lines[ref_start:]:
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"appendix|acknowledg(e)?ments?", line, flags=re.IGNORECASE):
            break
        if re.match(r"^\[?\d+\]?", line) or re.match(r"^[A-Z][^.]+\(\d{4}\)", line):
            ref_lines.append(line)

    return ref_lines


def extract_pdf_metadata(
    pdf_path: Path,
    parser: str = "auto",
    grobid_url: str = "http://localhost:8070",
) -> Tuple[str, str, List[str]]:
    """Extract (title, abstract, references) with parser strategy.

    parser:
      - auto: try GROBID then fallback to heuristic PyMuPDF parsing
      - grobid: require GROBID parse success
      - heuristic: only PyMuPDF+heuristic parsing
    """
    parser = parser.lower()
    if parser not in {"auto", "grobid", "heuristic"}:
        raise ValueError("parser must be one of: auto, grobid, heuristic")

    if parser in {"auto", "grobid"}:
        grobid_data = parse_pdf_with_grobid(pdf_path, grobid_url)
        if grobid_data is not None:
            return grobid_data
        if parser == "grobid":
            raise PDFExtractionError("GROBID parsing failed; check grobid_url/service health")

    pdf_text = extract_text_from_pdf(pdf_path)
    return parse_title_abstract_references(pdf_text)


def _normalize_reference_title(reference_line: str) -> str:
    cleaned = re.sub(r"^\[?\d+\]?\s*", "", reference_line).strip()
    chunks = [chunk.strip() for chunk in cleaned.split(".") if chunk.strip()]

    if len(chunks) >= 2:
        candidate = chunks[1]
    elif chunks:
        candidate = chunks[0]
    else:
        candidate = cleaned

    return re.sub(r"\s+", " ", candidate).strip()


def fetch_semantic_scholar_paper(title_query: str, timeout_s: int = 15) -> Optional[Dict]:
    encoded_query = urllib.parse.quote(title_query)
    fields = urllib.parse.quote("paperId,title,abstract,authors")
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={encoded_query}&limit=1&fields={fields}"
    )
    payload = _request_json(url, timeout_s=timeout_s)
    if not payload:
        return None
    data = payload.get("data") or []
    return data[0] if data else None


def fetch_semantic_scholar_author_profile(author_id: str, timeout_s: int = 15) -> Optional[Dict]:
    if not author_id.isdigit():
        return None
    fields = urllib.parse.quote("name,homepage,affiliations")
    url = f"https://api.semanticscholar.org/graph/v1/author/{urllib.parse.quote(author_id)}?fields={fields}"
    return _request_json(url, timeout_s=timeout_s)


def fetch_openalex_author_profile(author_name: str, timeout_s: int = 15) -> Optional[Dict]:
    if not author_name:
        return None
    q = urllib.parse.quote(author_name)
    url = f"https://api.openalex.org/authors?search={q}&per-page=1"
    payload = _request_json(url, timeout_s=timeout_s)
    if not payload:
        return None
    results = payload.get("results") or []
    if not results:
        return None
    top = results[0]
    affiliation = ""
    homepage = ""
    institution = top.get("last_known_institution") or {}
    affiliation = (institution.get("display_name") or "").strip()
    homepage = (institution.get("homepage_url") or "").strip()
    return {
        "name": top.get("display_name") or author_name,
        "affiliation": affiliation,
        "homepage": homepage,
    }


def build_reviewer_pool_from_references(reference_lines: Sequence[str], max_references: int = 30) -> List[ReviewerRecord]:
    reviewer_map: Dict[str, ReviewerRecord] = {}

    for ref_line in reference_lines[:max_references]:
        query_title = _normalize_reference_title(ref_line)
        if not query_title:
            continue

        paper = fetch_semantic_scholar_paper(query_title)
        if not paper:
            continue

        paper_title = (paper.get("title") or "").strip()
        paper_abstract = (paper.get("abstract") or "").strip()
        if not paper_title:
            continue

        paper_record = PaperRecord(
            paper_id=paper.get("paperId") or query_title,
            title=paper_title,
            abstract=paper_abstract,
        )

        for author in paper.get("authors") or []:
            author_name = (author.get("name") or "").strip()
            if not author_name:
                continue
            author_id = str(author.get("authorId") or f"name::{author_name.lower()}").strip()

            if author_id not in reviewer_map:
                reviewer_map[author_id] = ReviewerRecord(
                    reviewer_id=author_id,
                    name=author_name,
                    affiliation="",
                    homepage="",
                    paper_count=0,
                    papers=[],
                )

            reviewer = reviewer_map[author_id]
            existing_ids = {p.paper_id for p in reviewer.papers}
            if paper_record.paper_id not in existing_ids:
                reviewer.papers.append(paper_record)
                reviewer.paper_count += 1

    return list(reviewer_map.values())


def enrich_reviewer_profiles(reviewers: Sequence[ReviewerRecord], source: str = "openalex") -> List[ReviewerRecord]:
    source = source.lower()
    if source not in {"none", "openalex", "semantic_scholar"}:
        raise ValueError("source must be one of: none, openalex, semantic_scholar")

    if source == "none":
        return list(reviewers)

    for reviewer in reviewers:
        try:
            profile = None
            if source == "semantic_scholar":
                profile = fetch_semantic_scholar_author_profile(reviewer.reviewer_id)
                if profile:
                    affs = profile.get("affiliations") or []
                    if affs:
                        reviewer.affiliation = ", ".join(affs[:2])
                    reviewer.homepage = (profile.get("homepage") or "").strip()
            else:
                profile = fetch_openalex_author_profile(reviewer.name)
                if profile:
                    reviewer.affiliation = (profile.get("affiliation") or reviewer.affiliation).strip()
                    reviewer.homepage = (profile.get("homepage") or reviewer.homepage).strip()
        except Exception:
            continue

    return list(reviewers)


class Specter2Embedder:
    def __init__(self, use_cuda: bool = False):
        try:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "SPECTER2 embedding requires transformers, adapters and torch to be installed."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_aug2023refresh_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_aug2023refresh_base")
        self.model.load_adapter(
            "allenai/specter2_aug2023refresh", source="hf", load_as="proximity", set_active=True
        )
        self.device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            ).to(self.device)
            with self._torch.no_grad():
                output = self.model(**inputs)
            batch_emb = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(batch_emb)

        if not embeddings:
            return np.zeros((0, 768), dtype=np.float32)

        return np.vstack(embeddings)


def _paper_to_text(title: str, abstract: str) -> str:
    return f"{title.strip()} [SEP] {abstract.strip()}".strip()


def build_reviewer_embeddings(
    reviewers: Sequence[ReviewerRecord],
    embedder: Specter2Embedder,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    reviewer_embeddings: Dict[str, np.ndarray] = {}
    for reviewer in reviewers:
        paper_texts = [_paper_to_text(p.title, p.abstract) for p in reviewer.papers]
        paper_texts = [t for t in paper_texts if t]
        if not paper_texts:
            continue

        vectors = embedder.encode_texts(paper_texts, batch_size=batch_size)
        mean_vector = vectors.mean(axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm == 0:
            continue
        reviewer_embeddings[reviewer.reviewer_id] = mean_vector / norm

    return reviewer_embeddings


def rank_reviewers(
    submission_title: str,
    submission_abstract: str,
    reviewers: Sequence[ReviewerRecord],
    reviewer_embeddings: Dict[str, np.ndarray],
    embedder: Specter2Embedder,
    top_n: int = 10,
) -> List[Dict]:
    submission_vec = embedder.encode_texts([_paper_to_text(submission_title, submission_abstract)])[0]
    submission_norm = np.linalg.norm(submission_vec)
    if submission_norm == 0:
        raise ValueError("Submission embedding has zero norm")
    submission_vec = submission_vec / submission_norm

    rows = []
    for reviewer in reviewers:
        vec = reviewer_embeddings.get(reviewer.reviewer_id)
        if vec is None:
            continue
        score = float(np.dot(submission_vec, vec))
        rows.append(
            {
                "reviewer_id": reviewer.reviewer_id,
                "name": reviewer.name,
                "affiliation": reviewer.affiliation,
                "homepage": reviewer.homepage,
                "paper_count": reviewer.paper_count,
                "score": score,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]


def save_reviewer_pool_json(reviewers: Sequence[ReviewerRecord], output_path: Path) -> None:
    payload = []
    for reviewer in reviewers:
        payload.append(
            {
                "reviewer_id": reviewer.reviewer_id,
                "name": reviewer.name,
                "affiliation": reviewer.affiliation,
                "homepage": reviewer.homepage,
                "paper_count": reviewer.paper_count,
                "papers": [paper.__dict__ for paper in reviewer.papers],
            }
        )

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pdf_matching(
    pdf_path: Path,
    output_path: Path,
    top_n: int = 10,
    max_references: int = 30,
    pdf_parser: str = "auto",
    grobid_url: str = "http://localhost:8070",
    enrich_source: str = "openalex",
) -> Tuple[Dict, List[ReviewerRecord]]:
    title, abstract, references = extract_pdf_metadata(pdf_path, parser=pdf_parser, grobid_url=grobid_url)

    reviewers = build_reviewer_pool_from_references(references, max_references=max_references)
    if not reviewers:
        raise RuntimeError("No reviewers could be built from PDF references")

    reviewers = enrich_reviewer_profiles(reviewers, source=enrich_source)

    embedder = Specter2Embedder(use_cuda=False)
    reviewer_embeddings = build_reviewer_embeddings(reviewers, embedder)
    ranked = rank_reviewers(title, abstract, reviewers, reviewer_embeddings, embedder, top_n=top_n)

    response = {
        "submission": {"title": title, "abstract": abstract},
        "reference_count": len(references),
        "reviewer_pool_size": len(reviewers),
        "enrich_source": enrich_source,
        "pdf_parser": pdf_parser,
        "top_reviewers": ranked,
    }
    output_path.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")
    return response, reviewers


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Match a PDF submission to candidate reviewers")
    parser.add_argument("--pdf", required=True, help="Path to submission PDF")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--top-n", type=int, default=10, help="Top reviewer count")
    parser.add_argument("--max-references", type=int, default=30, help="Maximum references to use")
    parser.add_argument("--pdf-parser", default="auto", choices=["auto", "grobid", "heuristic"])
    parser.add_argument("--grobid-url", default="http://localhost:8070", help="Base URL of a GROBID service")
    parser.add_argument(
        "--enrich-source",
        default="openalex",
        choices=["none", "openalex", "semantic_scholar"],
        help="Reviewer profile enrichment source",
    )
    parser.add_argument(
        "--save-reviewer-pool",
        default="",
        help="Optional path to dump generated reviewer pool JSON",
    )

    args = parser.parse_args(argv)
    result, reviewers = run_pdf_matching(
        pdf_path=Path(args.pdf),
        output_path=Path(args.output),
        top_n=args.top_n,
        max_references=args.max_references,
        pdf_parser=args.pdf_parser,
        grobid_url=args.grobid_url,
        enrich_source=args.enrich_source,
    )

    if args.save_reviewer_pool:
        save_reviewer_pool_json(reviewers, Path(args.save_reviewer_pool))

    print(json.dumps({"status": "ok", "top_reviewers": len(result["top_reviewers"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
