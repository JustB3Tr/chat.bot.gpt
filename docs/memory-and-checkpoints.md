# Memory and Checkpoints

## Memory Strategy
- **Short-term buffer:** Retain the last N turns (token-budget aware). Used for immediate coherence.
- **Conversation summarizer:** Rolling summaries compressed every M turns to reduce context size while preserving intent and entities.
- **Long-term store:** Semantic vectors in pgvector/Qdrant keyed by user/session, with metadata (topics, referenced URLs/files, task types), plus **user preference embeddings** for tone/style. Retrieval pulls top-K plus recency-weighted items.
- **Entity store:** Lightweight table for canonical entities (people, projects, repos) to keep names/pronouns consistent, with **user persona facets** (tone, verbosity, formality, interests).
- **Artifacts cache:** Links to generated images/meshes and code snippets with fingerprints for reuse.

## Checkpoints
- **Definition:** A checkpoint is a named snapshot containing: conversation summary, latest user goals, selected entities, retrieved docs, and pending tasks.
- **Operations:**
  - `save_checkpoint(name, notes?)` → pins current snapshot.
  - `load_checkpoint(name)` → restores snapshot into active memory and surfaces notes to the model.
  - `list_checkpoints()` → returns available names + timestamps.
  - `delete_checkpoint(name)` → optional cleanup.
- **Storage layout:** Stored in relational DB rows with JSONB payloads; large docs referenced by vector IDs to avoid duplication.
- **Client UX:** UI exposes a dropdown/command palette to select or save checkpoints; CLI flag `--checkpoint <name>`. Allow **user persona selection** (e.g., “concise”, “playful”, “formal”) and **manual preference notes** to be pinned into checkpoints.

## Retrieval + Web Search Flow
1. Decide if external context is required (classifier or heuristic: unknown entities, stale data, news-like queries).
2. Issue web search via provider; fetch top results with scraper that strips boilerplate and enforces allow/deny lists.
3. Chunk, embed, and store fetched docs with provenance (URL, title, timestamp).
4. At response time, merge:
   - Short-term buffer
   - Rolling summary
   - Retrieved docs (RAG)
   - Checkpoint snapshot (if provided)
   - **User preference/persona vectors** to steer tone, verbosity, and stylistic choices
5. Track citations/provenance for transparency in UI.

## Persistence & Scaling
- **PostgreSQL** for checkpoints, entity store, and metadata; **Redis** for hot buffers.
- **Background jobs** to age out stale memories, rebuild summaries, and re-embed changed checkpoints.
- **Multi-tenant isolation:** Namespace user/org IDs in vector and relational stores; encrypt sensitive fields at rest.

## Evaluation Hooks
- **Memory regression tests:** Synthetic dialogs to ensure checkpoint restore injects correct entities/goals.
- **Latency metrics:** Track retrieval + embedding timing; alert on p95 spikes.
- **Quality probes:** Evaluate answers with/without checkpoints to verify uplift.
