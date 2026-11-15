"""Report which English/Persian page pairs are structurally out of sync.

The two editions share file names (`notebooks/X.ipynb` <-> `fa/X.ipynb`), and the
code cells are copied verbatim, so a cheap structural comparison catches most
divergence: a translated page that gained a section, a code cell or an image
since its Persian twin was written.

Usage:  python tools/check_bilingual_sync.py [--list-all]
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FA = ROOT / "fa"


def fa_path(en_path):
    rel = en_path.relative_to(ROOT).as_posix()
    if rel.startswith("notebooks/"):
        rel = rel[len("notebooks/") :]
    return FA / rel


def profile(path):
    """Comparable structure of one page."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        markdown = [c for c in cells if c["cell_type"] == "markdown"]
        code = [c for c in cells if c["cell_type"] == "code"]
        return {
            "markdown": len(markdown),
            "code": len(code),
            "code_text": "\n".join("".join(c["source"]) for c in code),
            "images": sum("".join(c["source"]).count("img/") for c in markdown),
            "headings": sum(
                len([l for l in "".join(c["source"]).splitlines() if l.startswith("#")]) for c in markdown
            ),
        }
    text = path.read_text(encoding="utf-8")
    return {
        "markdown": len([p for p in text.split("\n\n") if p.strip()]),
        "code": 0,
        "code_text": "",
        "images": text.count("img/"),
        "headings": len([l for l in text.splitlines() if l.startswith("#")]),
    }


def main():
    def is_page(path):
        if path.suffix not in {".ipynb", ".md"}:
            return False
        skipped = {"img", "_build-V1", ".ipynb_checkpoints"}
        return not any(part in skipped or part.startswith("_build") for part in path.parts)

    english = [p for p in sorted(ROOT.glob("notebooks/**/*")) if is_page(p)]
    english += sorted(p for p in ROOT.glob("course-fall-2026/**/*.md"))

    out_of_sync, missing, in_sync = [], [], []
    for en in english:
        fa = fa_path(en)
        if not fa.exists():
            missing.append(en.relative_to(ROOT).as_posix())
            continue
        a, b = profile(en), profile(fa)
        reasons = []
        if a["markdown"] != b["markdown"]:
            reasons.append(f"markdown cells {a['markdown']} vs {b['markdown']}")
        if a["code"] != b["code"]:
            reasons.append(f"code cells {a['code']} vs {b['code']}")
        elif a["code_text"] != b["code_text"]:
            reasons.append("code cells differ")
        if a["headings"] != b["headings"]:
            reasons.append(f"headings {a['headings']} vs {b['headings']}")
        if a["images"] != b["images"]:
            reasons.append(f"images {a['images']} vs {b['images']}")
        if reasons:
            out_of_sync.append((en.relative_to(ROOT).as_posix(), reasons))
        else:
            in_sync.append(en.relative_to(ROOT).as_posix())

    print(f"## out of sync ({len(out_of_sync)})")
    for path, reasons in out_of_sync:
        print(f"- `{path}` — " + "; ".join(reasons))
    print(f"\n## no Persian version ({len(missing)})")
    for path in missing:
        print(f"- `{path}`")
    if "--list-all" in sys.argv:
        print(f"\n## in sync ({len(in_sync)})")
        for path in in_sync:
            print(f"- `{path}`")
    else:
        print(f"\n({len(in_sync)} pages match structurally; pass --list-all to see them)")


if __name__ == "__main__":
    main()
