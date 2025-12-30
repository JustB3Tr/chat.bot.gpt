# Custom Multimodal Chatbot Architecture

## Goals
- Provide high-quality conversational help for everyday tasks, coding assistance, image generation, and basic 3D modeling guidance.
- Enable live web retrieval to stay current and source relevant context.
- Offer strong multilingual code support with focus on Python, JavaScript/TypeScript, and HTML/CSS.
- Maintain conversational memory with user-invoked checkpoints to recall key facts.
- Ship as a simple-to-run stack with cloud/on-prem flexibility.

## High-Level System Diagram
- **Client:** Web UI (Next.js/React) or CLI. Connects via WebSocket/HTTPS for streaming tokens.
- **Gateway/API:** FastAPI or Node/Express service that handles auth, rate limits, and routes requests to the orchestrator.
- **Orchestrator/Agent Runtime:** Python service (e.g., LangGraph/LangChain) that manages tools, memory, and routing. Implements guardrails and tracing.
- **Model Backends:**
  - **LLM:** Fully custom pretrained base model (20–70B target) with downstream adapters/LoRAs for code, multimodal prompting, and retrieval-aware behaviors. Supports function/tool calling.
  - **Image Generator:** Custom-trained diffusion pipeline (e.g., SDXL-compatible architecture) served behind an internal inference API.
  - **3D Generator:** Custom-trained text-to-3D stack (e.g., DreamFusion-style score distillation + mesh simplifier) exposed through an internal service.
- **Retrieval & Web Search:** Web search proxy (SerpAPI/Brave/Exa) and document fetcher with scraper + sanitization. Uses vector store (Qdrant/PGVector) for semantic search and caching.
- **Memory:** Short-term buffer + long-term vector store keyed by user/session. Checkpoints are pinned memory snapshots accessible by name.
- **Storage:** Object store for assets (images/meshes), relational DB for auth/quotas/metadata.
- **Observability:** OpenTelemetry traces/metrics, structured logs, prompt/version tracking, red-team/feedback loop.

## Request Flow
1. Client sends message (optionally specifying checkpoint to load/save).
2. Gateway authenticates, enforces quotas, and forwards to orchestrator.
3. Orchestrator resolves context:
   - Loads short-term buffer + relevant long-term memories.
   - Restores user-selected checkpoint, if any.
   - Runs retrieval (web + internal docs) when needed.
4. Tool routing:
   - Code questions -> code-specialist LLM with tool access (repl, tests, repo introspection).
   - Image/3D -> sends prompt to generation models and returns assets/links.
   - General Q&A -> generalist LLM + retrieval.
5. Response streamed to client; memory updated; optional checkpoint saved. **User preference signals** (tone, verbosity, style, interests) are refreshed on each turn and stored for personalization.

## Key Components
- **Conversation Controller:** Handles turns, streaming, retry logic, safety filters.
- **Tooling Layer:**
  - Web search + scraper with domain allow/deny lists.
  - Code tools: sandboxed REPL, unit-test runner, repo browser.
  - Image generator client; 3D generator client + mesh simplifier.
- **Memory Layer:**
  - Short-term sliding window (last N messages or token budget).
  - Long-term vector store with metadata (intent, entities, files referenced) and **user preference embeddings** (tone/style/verbosity/interests).
  - Checkpoints = named snapshots of (conversation summary, entities, retrieved docs, tool outputs).
- **Training/Eval:**
  - Datasets for general chat, coding, image prompt design, 3D prompt patterns, tool-use traces.
  - Evaluation harness with unit-style tests, tool-use success metrics, and human-rated samples.
- **Deployment:** Containerized services with IaC (Terraform), Helm charts for k8s; optional single-node docker-compose for simplicity.

## Minimal Runnable Stack (MVP)
- **Backend:** Python + FastAPI + LangGraph orchestrator.
- **Frontend:** Next.js chat UI with checkpoint selector/saver.
- **Memory:** PostgreSQL (pgvector) for long-term memory, Redis for short-term cache/session buffer.
- **Search:** Brave/SerpAPI adapter with HTML boilerplate removal and URL safety checks.
- **Models:**
  - Custom LLM served via vLLM/TensorRT-LLM on A100/H100 nodes with quantized replicas for throughput.
  - Custom diffusion model (image) served on GPU workers with safety filters and prompt/negative-prompt controls.
  - Custom text-to-3D service producing meshes; automatic decimation + glTF export.

## Security & Safety
- API keys stored in secrets manager; per-user rate limits.
- Content safety filters on inputs/outputs; domain allowlist for scraping.
- Sandboxed code execution with resource/time limits.
- Audit logs for tool invocations and checkpoint restores/saves.

## Extensibility Points
- Plug-in tool registry for new capabilities (e.g., calendar, email, cloud drive).
- Model abstraction to swap base LLMs or add adapters (code-heavy, multilingual, vision).
- Memory policies configurable per org/user.
