# Model Strategy and Training Plan

## Baseline Model Selection
- Train a **fully custom base LLM** (20–70B class) with tool-calling capability—no reliance on third-party hosted bases.
- Layer **specialized adapters/LoRAs** for:
  - **Coding:** Python, JavaScript/TypeScript, HTML/CSS, and shell; leverage curated code datasets (HumanEval+/CodeContests/repo-level Q&A).
  - **Image prompt design:** Fine-tune on prompt-to-image feedback and style tokens.
  - **3D prompt patterns:** Adapter trained on successful text-to-3D prompts and mesh metadata.

## Custom Training Pipeline
1. **Data Curation**
   - General chat: safety-aligned dialogues, helpfulness tuning.
   - Coding: repo-level tasks with tool-use traces; self-play for debugging.
   - Web grounding: Q&A with citations and retrieval snippets.
   - Image/3D prompts: paired with outputs and quality scores.
2. **Supervised Fine-Tuning (SFT)**
   - Train adapters on curated datasets; maintain separate heads/checkpoints for code vs. general vs. multimodal routing.
3. **DPO/Reward Modeling**
   - Preference data on correctness, security, and style adherence; add penalties for hallucinated URLs/versions.
4. **Tool-Use Augmentation**
   - Synthetic traces for search, code execution, and checkpoint operations.
5. **Continuous Evaluation**
   - Benchmarks: HumanEval/MBPP for code, DocVQA/Hotpot for retrieval, image prompt quality via CLIP score, 3D via mesh similarity and human ratings.

## Inference-Time Routing
- **Router policy** chooses between:
  - Generalist LLM (short answers, everyday Q&A).
  - Code-specialist (tool-heavy tasks, repo Q&A, debugging).
  - Multimodal prompt engineer (image/3D generation requests).
- Use lightweight classifier on intent and required tools; fall back to generalist if ambiguous but escalate to retrieval.

## Tool & Search Integration
- Structured tool schemas for search, code exec, file ops, checkpoint save/load, and asset generation.
- Strict output validation (pydantic) before executing tools; guardrails for file/network access.

## Memory-Aware Prompting
- System prompt injects:
  - User profile + active checkpoint summary
  - Short-term buffer + retrieved snippets + citations
  - Tool-usage conventions and safety rails
- Enforce token budget via dynamic truncation and summarization.

## Deployment Options (custom stack)
- **LLM serving:** Custom model served via vLLM/TensorRT-LLM with quantized replicas for throughput; fully private weights.
- **Embeddings:** Custom embedding model trained alongside the base; served through the same stack.
- **Image:** Custom diffusion pipeline (SDXL-style architecture) exposed through an internal API with safety filters.
- **3D:** Custom text-to-3D pipeline with mesh simplifier (decimation + glTF export) behind an internal service.

## Roadmap
- Phase 1: MVP with custom base/embedding checkpoints, checkpoints/memory, web search, and code sandbox.
- Phase 2: Add fine-tuned adapters, retrieval-augmented coding, and structured evaluation.
- Phase 3: Optimize serving (quantization/distillation), add organization-level memory policies, and GPU autoscaling.
