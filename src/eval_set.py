"""Golden questions for retrieval-quality metrics.

Each case lists the source_id(s) that must appear in the top-k hits.
Keyword-heavy cases (T-codes) exercise BM25; paraphrases exercise HNSW.
"""
from __future__ import annotations

from typing import TypedDict


class EvalCase(TypedDict):
    query: str
    expected_source_ids: list[str]
    allowed_roles: list[str]
    system: str
    note: str


GOLDEN_CASES: list[EvalCase] = [
    {
        "query": "How do I post a vendor invoice using MIRO?",
        "expected_source_ids": ["FI-AP-001"],
        "allowed_roles": ["common", "finance"],
        "system": "both",
        "note": "paraphrase + T-code (hybrid should win)",
    },
    {
        "query": "transaction F110 vendor payment run",
        "expected_source_ids": ["FI-AP-002"],
        "allowed_roles": ["common", "finance"],
        "system": "both",
        "note": "keyword / T-code heavy",
    },
    {
        "query": "How do I create a purchase order in S/4HANA?",
        "expected_source_ids": ["PROC-PO-001"],
        "allowed_roles": ["common", "procurement"],
        "system": "both",
        "note": "semantic paraphrase of ME21N SOP",
    },
    {
        "query": "Goods receipt against PO with MIGO",
        "expected_source_ids": ["PROC-GR-001"],
        "allowed_roles": ["common", "procurement"],
        "system": "both",
        "note": "keyword T-code MIGO",
    },
    {
        "query": "Walk me through the SIOP monthly cycle.",
        "expected_source_ids": ["PLAN-SIOP-001"],
        "allowed_roles": ["common", "planning"],
        "system": "both",
        "note": "semantic / process name",
    },
    {
        "query": "IBP demand planning cycle",
        "expected_source_ids": ["PLAN-IBP-001"],
        "allowed_roles": ["common", "planning"],
        "system": "both",
        "note": "keyword acronym IBP",
    },
    {
        "query": "How do I reset my SAP password?",
        "expected_source_ids": ["COMMON-LOGIN-001"],
        "allowed_roles": ["common"],
        "system": "both",
        "note": "common-role visibility",
    },
    {
        "query": "FB50 general ledger journal entry",
        "expected_source_ids": ["FI-GL-001"],
        "allowed_roles": ["common", "finance"],
        "system": "both",
        "note": "keyword T-code FB50",
    },
    {
        "query": "Purchase requisition creation and approval",
        "expected_source_ids": ["PROC-PR-001"],
        "allowed_roles": ["common", "procurement"],
        "system": "both",
        "note": "semantic process name",
    },
    {
        "query": "MRP run MD01 procedure",
        "expected_source_ids": ["PLAN-MRP-001"],
        "allowed_roles": ["common", "planning"],
        "system": "both",
        "note": "keyword T-code MD01",
    },
]
