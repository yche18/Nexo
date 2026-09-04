# Hybrid RAG

Agents invoke `search_knowledge_base` on demand. The pipeline performs query rewriting, dual-channel retrieval, two-stage RRF fusion, independent reranking, neighboring-chunk expansion and evidence assembly. Agents use the returned evidence to answer the current question. Retrieval is not mandatory for every request, and internal chunk IDs do not have to appear in user-facing answers.

## Retrieval and configuration

- **Keyword channel:** Elasticsearch and IK are pinned to **8.15.3**. BM25 retrieves the top 20 candidates for the original query. The image definition is `config/elasticsearch/Dockerfile`, with a dedicated Elasticsearch data volume.
- **Vector channel:** ChromaDB retains its existing default embedding. The original query and up to three rewrites each retrieve 20 candidates; this pipeline does not replace the embedding model.
- **Two-stage RRF:** fuse vector-query rankings and retain 20 candidates, then fuse that ranking with BM25 at equal channel weight and retain 20. Deduplicate by `chunk_id`, using RRF constant 60. This avoids giving the vector channel multiple times the weight simply because it has more rewrites.
- **Independent reranking:** candidates are sent to an HTTP reranking endpoint. The default model name is `BAAI/bge-reranker-v2-m3`; this is request configuration, not automatic model deployment or download. The provider must support the selected model and the `results/index/relevance_score` response format.
- **Evidence assembly:** select core chunks, then retrieve one preceding and one following neighbor for at most three seeds. Neighbors must share document, version and section. Deduplicate and retain complete chunks and IDs within a default **8000-character serialized JSON evidence budget**. This is neither a token count nor the whole request's context limit.

Relevant `.env` settings:

| Setting | Default / purpose |
| --- | --- |
| `ELASTICSEARCH_URL` | `http://elasticsearch:9200`, internal Compose address |
| `ELASTICSEARCH_INDEX` | `nexo_knowledge` |
| `RAG_CANDIDATE_K` / `RAG_RRF_K` | `20` / `60` |
| `RAG_NEIGHBOR_WINDOW` / `RAG_EXPANSION_SEEDS` | `1` / `3` |
| `RAG_EVIDENCE_MAX_CHARS` | `8000` |
| `RERANK_URL` | Full reranking endpoint; empty disables reranking |
| `RERANK_API_KEY` | Provider credential; keep it out of reports |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `RERANK_TIMEOUT_SECONDS` | `8` |

A timeout, failed request or invalid reranker response retains RRF order and records `failed`. An unconfigured endpoint records `disabled`. If one retrieval channel fails, the other may still return evidence; error messages are not treated as knowledge. Inspect `tool_calls[].retrieval` at `/trace/tool/{request_id}` for channel status, reranking status and actual evidence. A successful HTTP response alone does not prove the entire pipeline succeeded.

## New databases and legacy migration

Run the following commands from the Nexo project root.

**New database:** configure `.env`, then run `docker compose up -d --build`. The backend calls `initialize_keyword_index()` to synchronize existing `ready` knowledge chunks into a new or empty keyword index. If Elasticsearch is not ready, startup records degraded operation. Once it is ready, restart the backend and inspect actual tool traces.

**Legacy database:** chunks without version metadata remain available for read-only vector retrieval, but require explicit migration before further ingestion and full hybrid retrieval. Pause knowledge writes and back up the target Chroma `knowledge_base` collection, including IDs, documents, metadata and embeddings. Verify the target instance and keep the backup separate from the original database.

```sh
docker compose build backend
docker compose up -d --build elasticsearch
docker compose stop backend
docker compose run --rm --no-deps backend python -m mcp.rebuild_knowledge_index --migrate-legacy
docker compose up -d backend
```

Migration preserves legacy chunk IDs, text and boundaries while adding metadata and rebuilding the Elasticsearch index. It only processes the knowledge collection, not episodic memory or user profiles. By default, a remote Chroma connection failure cannot silently redirect migration to a local fallback. For an already migrated database, omit `--migrate-legacy` to rebuild only the keyword index. Restart the backend after an external rebuild or migration to refresh process-local metadata and retrieval caches.

Ingestion uses the existing knowledge APIs and does not report complete success until both stores finish. A failed batch may already have published some documents; the same input can be retried. This is not a cross-database transaction. Do not delete an entire Chroma volume to rebuild the knowledge index.

## Knowledge sources and expansion data

Import documents through the workbench, `POST /knowledge/add`, or `POST /knowledge/upload`. JSON uploads accept an array of documents. Each document needs `title` and `content` and can include metadata such as `source` and `doc_id`. The existing 500-character chunking logic is retained. ChromaDB handles vectorization and vector storage; Elasticsearch indexes the same chunks for keyword retrieval.

The [expanded knowledge collection](../backend/knowledge/it_support_expanded_v1/README.md) contains 32 demo manuals covering VPN, DNS, Wi-Fi, software, account access, devices and safety boundaries. Its `documents.json` imports each manual as three separate sections, producing 96 chunks. Together with the eight retained chunks in the frozen evaluation corpus, this gives 40 source manuals and 104 chunks. It is not the default count for every fresh deployment.

Preserve titles, sources and IDs when uploading `documents.json`. Sections are independent document units, so neighbor expansion does not cross section boundaries. This evaluation corpus cannot establish the benefits of expansion on long documents.

## Failures and inspection

The workbench knowledge-search interface displays candidates and scores. To inspect a real conversation, use its request's tool trace; a separate search cannot substitute for the evidence actually delivered to the agent.

- An unset endpoint produces `disabled` reranking. Configuring an endpoint does not deploy a service or guarantee support for the model.
- One channel may return results while another fails. Check `degraded`, channel status and error details together.
- Failed ingestion can leave partially published documents. Inspect the database before retrying identical input; do not delete the full ChromaDB volume, which may also hold memory collections.
- Technical, Network and Device Skills prioritize applicable SOPs. Retrieving a document does not guarantee that the answer preserves every required step; answer evaluation checks that separately.

See [Evaluation](evaluation.md) for dataset preparation, commands, Recall@K and answer-quality definitions.
