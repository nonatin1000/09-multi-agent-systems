# OpsPilot (Python)

Copiloto de plantão de produção com múltiplas estratégias de raciocínio (**ReAct**,
**plan-and-execute**, **reflect**) e **modo equipe** (supervisor + papéis especializados
sobre um blackboard compartilhado) construído com **FastAPI + LangGraph**. Convertido do
exercício original em TypeScript (`modulo04-criacao-de-agentes-autonomos-novo/09-multi-agent-systems`,
projeto "OpsPilot" em Node/Express).

> Snapshot de origem: a unidade 09 é o **estado de trabalho mais recente** do projeto TS
> original (cumulativo das 9 unidades do módulo — reasoning, memória, contexto, LangGraph,
> observabilidade e modo equipe).

## Arquitetura

```
POST /chat
  └─▶ contexto (histórico + memórias + resumo)
       └─▶ roteador (classifica: react | planExecute | reflect | team)
            ├─▶ react           (ReAct clássico com tools)
            ├─▶ planExecute     (planner → executor → replanner, LangGraph)
            ├─▶ reflect         (qualquer estratégia + crítico LLM em loop)
            └─▶ team            (supervisor delega a analista/planejador/executor)
                 └─▶ resposta (grava histórico, persiste auditoria, loga)
```

Camadas (espelham `src/` do TS, em `snake_case`):

| Pacote | Responsabilidade |
|---|---|
| `app/domain` | Tipos (dataclasses + `Protocol`), erros, normalização de severidade |
| `app/store` | `OpsStore`, `ConversationStore`, `RequestStore`, `ApprovalStore` — SQLite e in-memory |
| `app/memory` | Embeddings semânticos (sentence-transformers), memória de longo prazo, aprendizado |
| `app/context` | Estimativa de tokens e `ContextBuilder` (teto por seção do prompt) |
| `app/llm` / `app/agents/model.py` | Fábrica de modelo resiliente (retry → fallback → `ModelUnavailableError`) |
| `app/agents` | Tools LangChain, prompt de sistema, estratégia ReAct, registro de estratégias |
| `app/strategies` | `plan-and-execute` (LangGraph) e `reflect` (camada de crítico) |
| `app/team` | Modo equipe: supervisor, papéis (analista/planejador/executor), blackboard, grafo |
| `app/graph` | Grafo de produção (roteador + orquestração das estratégias) |
| `app/chat` | Sumarização de histórico, execução de turno "standalone" |
| `app/http` | API FastAPI (`/chat`, `/approvals`, `/requests`, `/stats`, `/memories`) |
| `app/mcp` | Servidor MCP (`list_alerts`, `open_incident`, `resolve_incident`) via stdio |
| `app/obs` / `app/trace` | Logger estruturado, estatísticas de custo, construção de trace |

## Setup

```bash
poetry install
cp .env.example .env
# Preencha OPENROUTER_API_KEY (e opcionalmente OPENROUTER_MODEL/OPENROUTER_MODEL_FALLBACK)
python -m app.main
# ou: uvicorn app.main:build_app --factory
```

MCP (stdio):

```bash
python -m app.mcp.server
```

## Testes

```bash
pytest        # roda com fakes — não requer OPENROUTER_API_KEY nem download de modelo
ruff check .
```

A suíte usa `FakeEmbedder` (vetores determinísticos, sem baixar o modelo de embeddings)
e estratégias/roteador fake nos testes de HTTP — mesmo padrão de "passa com fakes" do
projeto TS original.

## Notas de fidelidade (divergências propositais desta conversão)

- **Contrato JSON em `snake_case`** (convenção Python/FastAPI) onde o TS original usava
  `camelCase` (ex.: `llm_calls` em vez de `llmCalls`). Única divergência de contrato de API.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via `sentence-transformers`
  (mesmo modelo, mesma dimensão 384) no lugar de `@huggingface/transformers` — a contagem
  de rodadas em JS usava a versão WASM/transformers.js; aqui roda nativamente via PyTorch.
- **`AsyncLocalStorage` → `contextvars`**: `model-telemetry` e `chat-user-context` usam
  `contextvars.ContextVar`, o equivalente idiomático em Python para estado por-tarefa
  assíncrona.
- **`node:sqlite` → `sqlite3`** da stdlib — mesmo esquema de tabelas, mesmas regras de
  negócio (dedupe, recall top-k, auditoria de trace, CHECK constraints de severidade).
- **`web/` (React + Vite, "war room")** não foi portado — é uma SPA TypeScript sem
  equivalente natural em Python; o backend (`/chat`, `/requests/:id`, `/stats`) continua
  disponível para qualquer front-end consumir.
- **`arena.ts` / `bench.ts`** (harness de comparação/benchmark de estratégias, scripts de
  desenvolvimento) não foram portados — não fazem parte da API de produção.
- **Zod → Pydantic v2**: schemas de tools, request/response HTTP e saídas estruturadas de
  LLM (`withStructuredOutput`) usam `pydantic.BaseModel` no lugar de `zod`.
- **`@langchain/langgraph` → `langgraph` (Python)**: os grafos (`production-graph`,
  `plan-execute`, `team-graph`) usam `StateGraph` com `TypedDict` + `Annotated[..., operator.add]`
  no lugar de `Annotation.Root`; a semântica dos reducers (concatena trace/blackboard,
  soma `llm_calls` no modo equipe, sobrescreve o resto) foi preservada 1:1.

## Variáveis de ambiente

Ver `.env.example`. As mesmas do TS original: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`,
`OPENROUTER_MODEL_FALLBACK`, `PORT`, `OPSPILOT_CORS_ORIGINS`, `OPSPILOT_DB`, mais os
budgets de contexto opcionais (`CONTEXT_BUDGET_SUMMARY`, `CONTEXT_BUDGET_HISTORY`,
`CONTEXT_BUDGET_MEMORIES`).
