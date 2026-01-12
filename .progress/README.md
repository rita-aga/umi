# UMI Progress Tracking

This directory tracks the work needed to make UMI production-ready.

## Documents

| File | Purpose |
|------|---------|
| `production-readiness.md` | Comprehensive plan with all phases, tasks, and implementation details |
| `checklist.md` | Quick reference checklist for tracking completion |

## Current Status

**Phase**: Pre-implementation  
**Blockers**: None  
**Next Action**: Begin Phase 1 (Embedding Foundation)

## Quick Links

- [Full Plan](./production-readiness.md)
- [Checklist](./checklist.md)
- [Architecture Decisions](../docs/adr/)

## Summary of Gaps

1. **No embedding generation** - The `Entity.embedding` field is never populated
2. **No semantic search** - All retrieval is text substring matching
3. **VectorBackend exists but unused** - Cosine similarity code isn't wired up
4. **"Dual retrieval" is misleading** - Currently just multiple text searches
5. **Storage backends lack vector search** - LanceDB/PostgreSQL do text matching only

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Memory                                    │
│  remember() ──────────────────────────────────── recall()       │
│       │                                              │           │
│       ▼                                              ▼           │
│  ┌─────────────┐                           ┌──────────────────┐ │
│  │ EntityExtract│                           │  DualRetriever   │ │
│  │   (LLM)     │                           │                  │ │
│  └──────┬──────┘                           │  ┌────────────┐  │ │
│         │                                  │  │ Text Search│  │ │
│         ▼                                  │  └─────┬──────┘  │ │
│  ┌─────────────┐                           │        │         │ │
│  │  Embedder   │ ◄───────────────────────► │  ┌────────────┐  │ │
│  │ (OpenAI)   │    Query embedding         │  │Vector Search│  │ │
│  └──────┬──────┘                           │  └─────┬──────┘  │ │
│         │                                  │        │         │ │
│         │ Entity embeddings                │   RRF Merge      │ │
│         ▼                                  └────────┼─────────┘ │
│  ┌─────────────────────────────────────────────────┼──────────┐ │
│  │              StorageBackend                      │          │ │
│  │  store_entity() ◄── embedding ──► search_vector()          │ │
│  │                                                             │ │
│  │  [SimStorage] [LanceDB] [PostgreSQL+pgvector]              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Estimated Timeline

| Sprint | Weeks | Focus |
|--------|-------|-------|
| 1 | 1-2 | Embedding infrastructure |
| 2 | 3-4 | True dual retrieval |
| 3 | 5-6 | Production storage backends |
| 4 | 7-8 | Polish & documentation |
