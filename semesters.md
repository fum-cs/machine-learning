# Semesters

One living repo. Snapshot each offering with a git tag; do not rename the repo.

| Tag | Offering | Role | Notes |
|---|---|---|---|
| `2025-spring` | Spring 2025 | Materials used as part of **Learning Theory** | Frozen before Fall 2026 rebrand. Book title was “Machine Learning”. |
| *(untagged)* | Fall 2026 | Standalone **Machine Learning** | First offering as the main ML course (Soft Computing & AI). |

## End-of-term checklist

1. Update this table (one row: tag, term, role, what changed).
2. Commit the term’s final edits on `main`.
3. Tag and push:
   ```
   git tag -a YYYY-term -m "Snapshot: ..."
   git push origin YYYY-term
   ```
   Use tags like `2025-spring`, `2026-fall`.
4. Publish the book as usual (`jupyter-book build` + `ghp-import`).

## Links

- Current book (always latest): https://fum-cs.github.io/machine-learning/
- Frozen source for a past term: `https://github.com/fum-cs/machine-learning/tree/<tag>`
- GitHub Pages serves the latest `gh-pages` build only — it cannot show a past tag’s HTML at the same URL.
