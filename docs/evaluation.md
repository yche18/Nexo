# Evaluation

Nexo evaluates intent classification, dialogue behavior, retrieval recall and answer quality separately. Built-in frontend evaluation targets business regression; the standalone RAG runner uses a frozen corpus and per-question annotations. Their scores are not interchangeable.

## Datasets and entry points

| Dataset / entry point | Scope |
| --- | --- |
| [builtin_v1.json](../backend/evaluation/datasets/builtin_v1.json) | Default frontend suite: 11 intent inputs and six dialogue groups. Reports contain an intent aggregate and individual dialogue turns, not 17 independent scenarios |
| [rag_expanded_v1.json](../backend/evaluation/datasets/rag_expanded_v1.json) | Expanded RAG suite: 80 questions and 104 frozen chunks; 66 answerable and 14 unanswerable questions |
| [rag_expanded_v1_smoke10.json](../backend/evaluation/datasets/rag_expanded_v1_smoke10.json) | Ten fixed questions for a small initial validation and cost check |
| [rag_expanded_v1_annotations.json](../backend/evaluation/datasets/rag_expanded_v1_annotations.json) | Annotation notes and development/holdout split for the 48 new questions |
| [run_rag_eval.py](../backend/evaluation/run_rag_eval.py) | Calls the real chat endpoint, reads tool traces and optionally evaluates standalone search recall |
| [run_support_eval.py](../backend/evaluation/run_support_eval.py) | Historical support suite: 60 intent inputs and ten multi-turn scenarios using the earlier 14-class taxonomy |
| [run_harness_eval.py](../backend/evaluation/run_harness_eval.py) / [run_domain_eval.py](../backend/evaluation/run_domain_eval.py) | Targeted checks for harness continuity and domain routing |

The 80-question suite combines 32 retained questions with 48 new ones. The new set has 24 development and 24 holdout questions. The holdout questions have already been run and are no longer an unseen blind test. Documents and questions come from the same custom synthetic authoring process, without independent human labels or production traffic.

The corpus contains 40 source manuals: eight retained and 32 new. Each new manual has three independent section-level import units, yielding 96 new chunks plus eight retained chunks. Chunk counts are not manual counts. See the [knowledge collection guide](../backend/knowledge/it_support_expanded_v1/README.md).

## Metric definitions

### Standalone retrieval Recall@K

Each answerable question has a pre-annotated set of relevant chunk IDs:

```text
Question Recall@K = relevant chunks found in Top-K / all gold chunks for that question
Macro Recall@K = mean question-level Recall@K across answerable questions
Micro Recall@K = sum of relevant hits across questions / sum of gold chunks across questions
```

Unanswerable questions do not enter the recall denominator. Matching uses exact chunk IDs. Older annotations do not enumerate every equivalent SOP added later, so this is not exhaustive human-assessed semantic coverage. A separate `/search` call measures retrieval quality, not the agent's decision to search.

### Actual chat evidence recall

Take the union of chunks actually returned in the tool trace for a `/chat` request and compare it with that question's gold set. Results across multiple tool calls are deduplicated; this is not a single Top-K ranking. An answerable question with no knowledge call receives zero evidence recall. Missing traces are recorded as unavailable, not silently converted to zero.

### Answer quality and Judge output

The RAG Judge submits structured results through `submit_evaluation`, checking every required point, material errors and forbidden claims. A pass requires every necessary point with no material error or forbidden claim. Missing fields, wrong types and truncation produce invalid judgments. Invalid results remain in the denominator of all processed questions; no default score is substituted.

The built-in frontend Judge instead scores relevance, accuracy, completeness and helpfulness for the current support stage on a 0–1 scale, with a quality threshold of 0.75. Its `accuracy` is a model score, not a human correctness rate; `intent_accuracy` is the classification hit rate for the intent inputs. Invalid judgments are excluded from quality-score means. Baseline regression is computed only for comparable cases, models, judging contracts and runtime settings, with previous baselines archived separately.

## Recorded full 80-question run

Run date: **2026-11-13**. The run used `deepseek-flash` with thinking disabled, an isolated enforce service and the reranking service configured at that time. All questions ran once in order, with standalone search before chat. Failures were not rerun, and the result does not combine the best attempts from targeted tests.

| Metric | Result |
| --- | ---: |
| Questions processed | 80/80 |
| Standalone macro Recall@1 | 56.82% |
| Standalone macro Recall@3 | 84.09% |
| Standalone macro Recall@5 | 93.18% |
| Standalone micro Recall@5 | 90.625% |
| Actual chat evidence macro recall | 55.30% |
| Chats using knowledge retrieval | 43/80 |
| Automated Judge passes / all questions | 79/80 (98.75%) |
| Valid Judge results | 79/80 |
| Search / chat endpoint errors | 0 / 0 |
| Mean / maximum chat latency | 7.95s / 25.67s |

**98.75% is an automated pass rate, not human-verified answer accuracy.** Review found that `expanded_multi_04` (black screen and application crashes) passed despite missing required safety checks and saving work before restart. This review finding is retained alongside the automated result without overwriting the original judgment.

`expanded_n02` (VPN connected, project page returning 403, and a question about borrowing someone else's account) had missing or unexpected Judge point fields. It was the only invalid judgment. Successful endpoints did not mean degradation-free retrieval: three separate events were recorded, comprising a chat reranker timeout in `rag_16`, a chat neighbor-expansion KeyError in `expanded_n05`, and a standalone reranker timeout in `expanded_multi_05`.

### Interpretation and limits

- High standalone recall does not mean the agent searches every time. Tool selection contributes to the gap between standalone and chat evidence recall.
- Some passing answers did not use knowledge retrieval; the overall pass rate cannot be attributed entirely to RAG.
- The corpus is small and its sections are independent. These results do not establish large-corpus performance or gains from neighboring-chunk expansion.
- Cache and Monitor state were reused within the run. Search before chat may warm caches; the latency figures are not cold-start or concurrency benchmarks.
- The Judge and business models belong to the same model family. Structured output constrains format but does not prevent missed requirements.
- The original report's `limits` contains an old hardcoded “Seven-chunk” description. The actual frozen corpus has 104 chunks. Original reports remain unchanged.
- Results belong to the recorded working-tree snapshot, not automatically to later code. Historical 95.0% intent accuracy / 0.945 macro-F1 on the 14-class suite cannot be presented as results for the current 19-class taxonomy.

### Records and verification

This summary is derived from `live-report.json`, `summary.json` and `review.md` under the local directory `data/eval/rag_current_full80/20261113/`. Original answers, failures, degradation events, snapshots and usage records remain in the Git-ignored data directory. The public repository contains datasets, scripts and this summary. The hashes below allow holders of the original files to check them; they do not replace public raw reports or establish independent validation.

- Dataset version: `rag-expanded-v1-80`
- Dataset SHA256: `88fcfc7125f0e8d0ea6c491b484d19462c34d9599fe497f40178047b665c3017`
- Full report SHA256: `1eefdae4f3c3b517a71cea03d0d82fb5ee02d06aa7f188f08fbd30e0cf9ed0ea`
- Summary SHA256: `31f0b8500ab634f122b42eb3b3e2029f4324b23ec1fd07879fa67b87c4f83ddf`

## Running evaluation

Run these commands from the Nexo project root. Offline checks validate the dataset and metric/Judge schema logic without calling models:

```sh
python backend/evaluation/run_rag_eval.py --dataset backend/evaluation/datasets/rag_expanded_v1.json --validate-only
python backend/evaluation/run_rag_eval.py --dataset backend/evaluation/datasets/rag_expanded_v1.json --self-test
```

### Preparing a live environment

1. Use an isolated backend, Redis, ChromaDB instance and Elasticsearch index so evaluation does not write to ordinary conversations. Set `NEXO_HARNESS_MODE=enforce`, align the overall and per-agent models, and use the same thinking-disabled settings as the application.
2. Configure hybrid retrieval and a working reranking endpoint as described in [Hybrid RAG](rag.md). Changes to models, reranking or modes require a separately reported run.
3. Prepare a knowledge collection whose **IDs, titles, chunk indexes and text exactly match** the dataset's `corpus`. The runner checks this before any model calls and aborts on mismatch. Importing only the 96 new sections does not guarantee matching IDs for the eight retained historical chunks.
4. There is currently no public one-command script to restore this historical corpus into an arbitrary empty database. Restore an available matching backup into an isolated environment. For a different corpus, create a new dataset version and review gold labels; do not disable consistency checks or alter old IDs merely to obtain a score.
5. Explicitly select the expanded dataset. The runner still defaults to the earlier `rag_support_v2.json`; omitting the argument does not run the 80-question suite.

The following live commands invoke both business models and the Judge. They assume a prepared isolated service at `http://localhost:18009`. ChromaDB connection variables must point to the same isolated corpus; an ordinary service on that port is not an interchangeable substitute.

```sh
# Begin with the fixed ten-question set. Use a new report filename each time.
python backend/evaluation/run_rag_eval.py --dataset backend/evaluation/datasets/rag_expanded_v1_smoke10.json --api-base http://localhost:18009 --standalone-retrieval --application-revision YOUR_COMMIT_OR_WORKTREE_SNAPSHOT --report data/eval/rag_expanded/smoke-run1.json

# Review the complete small run before explicitly starting all 80 questions.
python backend/evaluation/run_rag_eval.py --dataset backend/evaluation/datasets/rag_expanded_v1.json --api-base http://localhost:18009 --standalone-retrieval --application-revision YOUR_COMMIT_OR_WORKTREE_SNAPSHOT --report data/eval/rag_expanded/full-run1.json
```

Without `--standalone-retrieval`, the runner still evaluates chat evidence and answers but does not separately measure search Recall@K. Existing report paths are rejected to prevent overwriting. Retain questions, answers, evidence, errors, Judge contracts and hashes for every attempt.

The RAG runner records per-question errors and may continue to subsequent questions. It does not guarantee automatic batch termination on insufficient balance or provider errors; the operator should stop the process promptly and retain the partial report. Do not assume it follows the targeted Harness runner's stop policy. Retrying invalid Judge outputs is a separate operation and must not be mixed into the original single-run result.

## Version preservation

Historical RAG datasets, intent datasets and evaluation scripts retain their original paths. Original labels and reports remain unchanged. Development diaries are stored in a local ignored directory; public guides describe the current design and explicitly dated runs. When publishing, include the new datasets and knowledge sources as well as the documentation; committing only the README does not include them.
