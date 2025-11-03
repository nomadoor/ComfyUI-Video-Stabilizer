# AGENTS.md — Codex Node Development (ComfyUI / Python-only)

**Motto**

> Small, clear, safe steps — grounded in real data and reproducible code.

---

## 🧭 Principles

- Prefer clarity over cleverness; explicit schemas over implicit behavior.
- One node = one purpose; no shared mutable state.
- Keep changes minimal, reviewable, reversible.
- Avoid new dependencies unless declared under `docs/requirements/`.
- Follow ComfyUI V3 schema for all nodes; align with official Comfy-Org docs.
- Outputs are deterministic (same input → same output).

---

## 🧩 Agent Overview

| Agent             | Role/Backend          | Description                                                                                     |
| :---------------- | :-------------------- | :---------------------------------------------------------------------------------------------- |
| `codex-nodegen`   | MCP (context7/serena) | Generates/updates V3-schema node files from specs in `docs/requirements/` into `nodes/`.       |
| `codex-docsync`   | same                  | Syncs schema metadata → README/Markdown tables; keeps docs in lockstep with code.              |
| `codex-validator` | same                  | Lints/validates nodes for schema compliance and safety; runs pre-commit via `codex validate`.  |

---

## 📚 Knowledge & References

- Official reference (primary):  
  - ComfyUI Development Overview — https://docs.comfy.org/development/overview
- Minimal dependencies with narrow version pins (e.g., `opencv-python-headless>=4.9,<4.11` when needed).

---

## ⚙️ Workflow

1) **Plan**  
   - Write the spec in `docs/requirements/feature-*.md` (purpose, inputs/outputs, behavior, acceptance, category).

2) **Generate**  
   - Command: codex generate node --from docs/requirements/feature-example.md
   - Outputs:
     - `nodes/<NodeName>_v3.py`  
     - (optional) `docs/generated/<NodeName>.md`

3) **Verify**  
    - `define_schema()` present  
    - Inputs/outputs match spec (types/order/defaults/widgets)  
    - Category/description align; no unsafe imports or side effects

4) **Implement & Test**  
    - Load in portable ComfyUI; exercise UI & runtime; confirm determinism  
    - If mismatch, update `docs/requirements/*.md` and regenerate

5) **Reflect**  
    - Append ADR: `docs/adr/YYYY-MM-DD-node-change.md`  
    - Update `CHANGELOG.md` (SemVer)

---

## 🧱 V3 Schema Essentials

- `name` / `description` / `category` (user-facing clarity)
- `inputs` (types, widgets, defaults) / `outputs` (types, order)
- Stable node id (avoid breaking existing graphs)
- Optional help text for ports/params
- No hidden randomness; document any non-deterministic behavior (avoid if possible)

---

## 🧪 Tests & Acceptance

- Unit tests: boundary values (min/max/empty), error messages, type checks, determinism
- Golden tests (media nodes): short fixtures with frame/hash checks
- Acceptance: must satisfy the `docs/requirements/*.md` criteria (I/O types, behavior, performance hints)

---

## 🔒 Safety & Side Effects

- No `eval` or dynamic code generation
- Network/file I/O only if specified in the feature doc
- Fail softly with clear messages; do not crash the graph
- Validate inputs early (type/range/None)

---

## 🧰 Coding Style & Limits

- ≲ 300 LOC per node; one class = one capability
- Key functions have docstrings (purpose, I/O, assumptions)
- Reused constants in `config.py` (optional)
- Prefer pure functions; avoid shared state

---

## 🤝 Collaboration & Escalation

- Escalate before merge if:
- Requirements are ambiguous or conflicting
- Potential compatibility breakage
- License/redistribution doubts (models/assets)
- Communicate when confidence < 80%; honesty > speed
- Unverified behavior defaults to no-op (fail-safe)

---

## ✅ Quick Checklist (Pre-Commit)

- [ ] Spec exists and is current (`docs/requirements/*.md`)
- [ ] `define_schema()` correct; I/O & descriptions match
- [ ] Deps pinned (narrow ranges) in `pyproject.toml` if used
- [ ] No undeclared side effects (network/I/O/exec)
- [ ] Unit & golden tests pass deterministically
- [ ] ADR + `CHANGELOG.md` updated (SemVer)
