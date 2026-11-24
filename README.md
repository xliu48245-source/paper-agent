# paper-agent

An AI research assistant that searches arXiv, reads abstracts, and compares papers for you.

Built from scratch **without any agent framework**. The tool-use loop is hand-written (~25 lines) so every decision the model makes is visible and debuggable.

## Demo

```
你：幫我找 3 篇關於 LLM hallucination detection 的論文，比較它們的方法
  [步驟 1] 呼叫 search_arxiv({"query": "LLM hallucination detection", "max_results": 5})
  [步驟 2] 呼叫 get_paper({"arxiv_id": "2506.09886v2"})
  [步驟 2] 呼叫 get_paper({"arxiv_id": "2603.22812v1"})
  [步驟 2] 呼叫 get_paper({"arxiv_id": "2608.18115v1"})

Agent：以下為從 arXiv 檢索到的 3 篇論文…（論文簡介 + 方法比較表）
```

What happened:
- **Step 1**: The model translated a Chinese question into English search keywords and over-fetched 5 candidates.
- **Step 2**: It picked 3 papers covering different approaches and fetched them with **parallel tool calls** in a single turn.

## How it works

```
user question
     │
     ▼
┌─────────────┐   wants a tool?   ┌──────────────┐
│     LLM     │ ────── yes ─────▶ │  run tool    │
│  (Gemini)   │ ◀── tool result ─ │ (arXiv API)  │
└─────────────┘                   └──────────────┘
     │ no
     ▼
final answer
```

| File | Role |
|---|---|
| `agent.py` | The agent loop, model calls, retries, CLI |
| `tools.py` | Tool implementations + JSON Schema declarations |

### Design decisions

- **Hand-written loop, SDK auto-calling disabled**: makes each step observable and keeps the logic portable across model providers.
- **Provider-agnostic tool declarations**: tools are plain JSON Schema dicts; only `build_gemini_tools()` is Gemini-specific, so switching to Claude or OpenAI means changing one function.
- **Errors go back to the model, not to the user**: a failing tool returns `{"error": ...}` so the model can retry with different arguments instead of crashing.
- **Guardrails**: a `MAX_STEPS` limit prevents infinite tool loops; `max_results` is clamped server-side rather than trusting model input; abstracts are truncated in search results to save tokens.
- **Rate-limit handling**: exponential backoff on HTTP 429/503 (the free tier hits these).

## Setup

Requires Python 3.10+ and a [Gemini API key](https://aistudio.google.com/) (free tier works).

```bash
git clone https://github.com/xliu48245-source/paper-agent.git
cd paper-agent
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY and GEMINI_MODEL
python agent.py
```

## Roadmap

- [x] v0.1 — hand-written tool-use loop, arXiv search & fetch
- [ ] v0.2 — save reports to Markdown, per-step timing, configurable thinking level
- [ ] v0.3 — conversation memory management, RAG over full-text PDFs
- [ ] v0.4 — evaluation set: measure success rate, steps and token cost; compare models
