# Custom AI Chatbot Blueprint

This repository outlines a custom-built, custom-trained multimodal AI chatbot focused on everyday help, strong coding assistance, image generation, and lightweight 3D modeling guidance. It is designed to support live web retrieval, conversational memory, and user-driven checkpoints.

## What you get
- High-level architecture for orchestrating a multimodal assistant with code, image, 3D, and web search tools.
- Memory and checkpoint design to maintain context without long prompts.
- Model strategy for baseline selection, fine-tuning, and inference-time routing.
- Security and deployment considerations for an MVP and a scalable follow-on.
- **New:** A training scaffold to pretrain a custom decoder-only LLM from scratch.

## Getting started
- Review the architecture overview in [`docs/architecture.md`](docs/architecture.md).
- See how memory, long-term storage, and checkpoints work in [`docs/memory-and-checkpoints.md`](docs/memory-and-checkpoints.md).
- Explore model choices, training, and routing in [`docs/model-strategy.md`](docs/model-strategy.md).
- Learn how to run tokenizer/model pretraining in [`docs/training-pipeline.md`](docs/training-pipeline.md).
- Check install options and offline/airgapped guidance in [`docs/training-pipeline.md`](docs/training-pipeline.md).

## Next steps
- Choose your initial stack (FastAPI + LangGraph for backend; Next.js for UI) and connect your **custom-trained** models for an MVP.
- Wire in web search + scraping with safety filters and a vector store (pgvector/Qdrant).
- Implement checkpoint endpoints (`save/load/list`) and expose them in the UI for quick recall.
- Add observability (OpenTelemetry, structured logs) and a basic evaluation harness for memory and tool-use quality.
