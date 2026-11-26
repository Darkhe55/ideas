# Ideas

[简体中文](README.md) | [English](README.en.md)

A repository of ideas, arguments, and final designs. Docs-first, with the essential code embedded in the documents.

## Layout

```
.
├── README.md          Chinese README
├── README.en.md       English README (this file)
└── docs/
    ├── 简略说明版/    Draft: up to 2 pages
    ├── 扩展中途版/    In progress: > 2 pages, ≤ 15 pages
    ├── 最终版/        Final: > 15 pages
    └── en/            English docs (same hierarchy)
        ├── brief/
        ├── extended/
        └── final/
```

The three stage folders are **three stages of one document**, not three documents. Chinese and English each keep one counterpart document.

## Single-copy rule

- **One document per language.** At any time an idea has exactly one document in each language; the three stages are never maintained in parallel.
- Drafts (≤ 2 pages) go in `简略说明版` (brief); once the document grows past 2 pages, `git mv` it to `扩展中途版` (extended); past 15 pages, move it to `最终版` (final).
- Move with `git mv`; keep the filename unchanged. Earlier stages remain in git history only.
- Update the document header's `Stage` field after moving.
- English docs live under `docs/en/` and mirror the Chinese stages: brief ↔ 简略说明版, extended ↔ 扩展中途版, final ↔ 最终版.

## Documents

| Idea | Stage | 中文 | English |
| --- | --- | --- | --- |
| Fast Prediction Framework | Final | [快速预测框架](docs/最终版/快速预测框架.md) | [Fast Prediction Framework](docs/en/final/fast-prediction-framework.md) |
| How Do I Know What I Don't Know | In progress | [我如何知道我不知道什么](docs/扩展中途版/我如何知道我不知道什么.md) | [How Do I Know What I Don't Know](docs/en/extended/how-do-i-know-what-i-dont-know.md) |

## Conventions

- Key code is PyTorch pseudocode: readable and reproducible first, runnable second.
- Use relative links; Chinese docs are primary, with English terms in parentheses at first mention; Chinese and English versions are kept in sync.
- New ideas start in `简略说明版` and move up as they grow.

## Security

This is a **public repository**. Never commit keys, tokens, passwords, personal data, or private datasets. `.gitignore` excludes common sensitive files; check before committing:

```bash
git status
git diff --cached
```

## Sync

```bash
git remote add origin ssh://git@ssh.github.com:443/Darkhe55/ideas.git
git push -u origin main
```
