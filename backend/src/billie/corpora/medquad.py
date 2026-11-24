"""MedQuAD -> passages.

MedQuAD (Ben Abacha & Demner-Fushman, 2019) has 47,457 QA pairs from 12 NIH sites.
Three MedlinePlus subsets (ADAM, Drugs, Herbs) ship with answers removed for
copyright reasons, so they contribute no passages.

One passage per answer chunk; the question is the passage title, which BM25
weights 2x (see ``PassageIndex.search_bm25``).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from billie.retrieval.base import Passage
from billie.retrieval.chunking import chunk_text

_SUBSET = re.compile(r"^\d+_(.+?)(?:_QA)?(?:_XML)?$")


def iter_medquad(root: Path, max_chars: int = 1200) -> Iterator[Passage]:
    for xml_path in sorted(root.rglob("*.xml")):
        try:
            doc = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        # The XML `id` attribute is not unique: "0000001" exists in ten subset folders, and
        # CancerGov reuses one id across files about different diseases. Subset + file name is.
        subset_no = xml_path.parent.name.split("_", 1)[0]
        doc_id = f"{subset_no}-{xml_path.stem}"
        url = doc.get("url", "")
        site = doc.get("source") or _subset_name(xml_path.parent.name)
        focus = (doc.findtext("Focus") or "").strip()
        semantic_group = (doc.findtext(".//SemanticGroup") or "").strip()
        # MedQuAD reuses `qid` across different questions in one document, so passage ids
        # are built from the document id and the pair's position instead.
        for position, pair in enumerate(doc.iter("QAPair"), start=1):
            question_node = pair.find("Question")
            question = " ".join((pair.findtext("Question") or "").split())
            answer = (pair.findtext("Answer") or "").strip()
            if not question or not answer:
                continue
            qid = f"{doc_id}-{position}"
            qtype = question_node.get("qtype", "") if question_node is not None else ""
            for n, chunk in enumerate(chunk_text(answer, max_chars=max_chars, overlap_chars=150)):
                yield Passage(
                    id=f"medquad:{qid}#{n}",
                    doc_id=qid,
                    title=question,
                    text=chunk,
                    source=f"MedQuAD · {site}",
                    url=url,
                    metadata={"focus": focus, "qtype": qtype, "semantic_group": semantic_group, "chunk": n},
                )


def iter_medquad_sample(path: Path) -> Iterator[Passage]:
    """The tiny JSON sample shipped for tests/demo (list of {id, question, answer})."""
    for item in json.loads(path.read_text(encoding="utf-8")):
        yield Passage(
            id=f"medquad-sample:{item['id']}",
            doc_id=str(item["id"]),
            title=item["question"],
            text=item["answer"],
            source="MedQuAD sample",
            metadata={"focus": item.get("focus", ""), "qtype": item.get("question_type", "")},
        )


def _subset_name(folder: str) -> str:
    match = _SUBSET.match(folder)
    return match.group(1).replace("_", " ") if match else folder
