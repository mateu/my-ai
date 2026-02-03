# Roadmap

This document tracks candidate improvements for the AI Memory system.

## Short-Term
- Expand CLI commands (`/import`, `/stats`, `/explain`) to improve observability and iteration speed.
- Define chat import strategy (what to extract, how to map roles/timestamps, how to handle implicit/explicit extraction).
- Add explicit memory schema validation and canonicalization (store `field`, `value`, `confidence`, `source`).
- Switch to real embeddings by default and persist vector data across sessions.
- Replace rule-based implicit trait inference with LLM-based extraction and consistent categories.

## Medium-Term
- Add explicit memory freshness tracking and conflict resolution (e.g., handle updates like name changes).
- Introduce topic tags for explicit facts (e.g., pets, work, preferences) and filter during recall.
- Expand CLI commands (`/export`, `/forget field`, additional import formats).
