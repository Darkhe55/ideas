# Documentation Conventions

[简体中文](../README.md) | [English](README.md)

## Lifecycle

One idea has **one document per language**, and it migrates between three folders as it grows:

| Stage | Folder | Length | Focus |
| --- | --- | --- | --- |
| Draft | `docs/简略说明版/` | ≤ 2 pages | problem, one-line idea, core structure, key metrics, main risks |
| In progress | `docs/扩展中途版/` | > 2 pages, ≤ 15 pages | derivations, alternatives, pseudocode, experiment plans, open questions |
| Final | `docs/最终版/` | > 15 pages | formalization, interfaces, key code, evaluation, limitations, related work, roadmap |

Rules:

1. **Single copy per language**: at any time an idea has exactly one document in each language; the three stages are never maintained in parallel.
2. **Move, don't copy**: use `git mv` to move the file; the filename stays unchanged.
3. **Update the stage field** in the header after moving.
4. **Page count**: estimate from A4 print/export pages (~40 lines/page; fewer when code and tables are dense).

## Chinese–English mirror

English documents live under `docs/en/` with the same stage hierarchy as the Chinese docs:

| Chinese | English |
| --- | --- |
| `docs/简略说明版/` | `docs/en/brief/` |
| `docs/扩展中途版/` | `docs/en/extended/` |
| `docs/最终版/` | `docs/en/final/` |

- English files use English filenames.
- Chinese and English versions are kept in sync; update both when the content changes.
- Cross-link the two versions with the `Language` line in the header.

## Migration flow

```
简略说明版 (draft, ≤ 2 pages)
      │ grows past 2 pages
      ▼
扩展中途版 (in progress, 2–15 pages)
      │ grows past 15 pages
      ▼
最终版 (final, > 15 pages)
```

Earlier stages remain in git history only.

## Header template

```markdown
# Idea Name

> Stage: Draft / In Progress / Final · Status: Draft / WIP / Stable vX.Y · Last updated: YYYY-MM-DD
> Language: [中文](path) · English

## ...
```

## Naming

- Chinese filenames use the idea name, no date prefix (dates live in the header).
- Keep the filename unchanged when moving between stages.
- English files use an English translation of the idea name.
