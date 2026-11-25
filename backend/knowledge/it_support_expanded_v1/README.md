# IT Support Knowledge Expansion v1

This collection adds 32 fictional enterprise IT manuals for Nexo demonstrations and retrieval evaluation. It supplies no real enterprise addresses, passwords, licenses, approval timelines or asset facts. The original eight knowledge documents are unchanged.

## Data layout

- `n01.md` through `n32.md` are the 32 source manuals. Each has three independent sections: applicability and diagnosis, troubleshooting steps, and verification and handoff.
- `documents.json` contains 96 section-level import units accepted by the existing `/knowledge/add` or `/knowledge/upload` interfaces. Each section is shorter than 500 characters and produces one chunk under the existing chunking logic. Chunk size, overlap and algorithms are unchanged; no filler is duplicated.
- Each section has a distinct `doc_id`, preventing whole-document version replacement from overwriting another section of the same manual. A shared `source` identifies the parent manual. Neighbor expansion does not cross these sections.
- `manifest.json` maps the 32 manuals to their 96 import units. The 40 source manuals (eight retained plus 32 new) and 104 indexed chunks are different counts; the 96 sections are not 96 independent manuals.

## Coverage

VPN 401/403/timeouts, intranet DNS, guest Wi-Fi, proxy 407, certificate warnings, account and MFA recovery, access scope/renewal/approval boundaries, Windows and macOS installation, licenses, application crashes, sync conflicts, HTTP 500, update rollback, email safety, no-power/black-screen issues, displays, printers, battery safety and demo asset lookup.

The manuals remain in Chinese because their exact text is part of the frozen retrieval corpus and evaluation evidence. Translating those files would create a different corpus and require a new dataset version; translating this guide does not change the runtime knowledge.

## Import and validation

Use the existing API to add `documents.json`. Do not replace the retained documents or delete/rebuild data volumes. Check the target environment even when a complete backup exists, so an isolated evaluation database is not confused with the ordinary service. Reimporting identical `doc_id` and content follows the existing ingestion logic.

Related datasets are `backend/evaluation/datasets/rag_expanded_v1.json` (80 questions), `rag_expanded_v1_smoke10.json` (10 fixed questions), and `rag_expanded_v1_annotations.json` (annotations and holdout split for the 48 new questions). The previous 32 questions and their reports are retained. The new questions and knowledge were authored in the same workflow; they are not an independently human-labeled or public benchmark.

See [Evaluation](../../../docs/evaluation.md) for metric definitions, recorded results and commands, and [Hybrid RAG](../../../docs/rag.md) for retrieval configuration.
