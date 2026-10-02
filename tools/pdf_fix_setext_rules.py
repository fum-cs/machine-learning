#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Find (and optionally fix) `---` lines that MyST parses as a *setext heading*.

CommonMark turns

    some text
    ---

into a level-2 heading whose title is "some text". In Jupyter Book v2 that
title becomes a LaTeX sectioning argument, so a heading that wraps an image
or a long paragraph produces

    ./chapter.tex:237: Paragraph ended before \\@sect was complete.

and xelatex exits 1 with no `!` line in the log. (The build runs xelatex with
`-file-line-error`, which rewrites `!` errors into `./file.tex:NN:` lines, so
grepping the log for `^!` finds nothing.)

Nearly all of these `---` lines are meant as a horizontal rule, so the fix is
to insert a blank line before them, restoring the intended thematic break.

Usage:
    python tools/pdf_fix_setext_rules.py            # report only
    python tools/pdf_fix_setext_rules.py --fix      # report and repair
"""
import io
import json
import os
import sys

RULES = ('---', '----', '-----')
SKIP_DIRS = ('_build', 'node_modules', 'venv', '.venv', '.freebuff')


def skip(name):
    return name in SKIP_DIRS or name.startswith(('_build', '.'))


def targets():
    for dirpath, dirnames, filenames in os.walk('.'):
        dirnames[:] = [d for d in dirnames if not skip(d)]
        for name in filenames:
            if name.endswith(('.md', '.ipynb')):
                yield os.path.normpath(os.path.join(dirpath, name))


def read(path):
    with io.open(path, encoding='utf-8', newline='') as f:
        return f.read()


def write(path, text):
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)


def find(path, label, lines):
    """Return the indexes of `---` lines that would become setext headings."""
    return [i for i in range(1, len(lines))
            if lines[i].strip() in RULES and lines[i - 1].strip()]


def fix_notebook(path, fix, found):
    raw = read(path)
    nb = json.loads(raw)
    changed = False
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'markdown':
            continue
        src = cell['source']
        for j in find(path, 'cell %d' % i, src):
            found.append((path, 'cell %d line %d' % (i, j + 1),
                          src[j - 1].strip()[:60], src[j].strip()))
            if fix:
                src.insert(j, '\n')
                changed = True
    if changed:
        write(path, json.dumps(nb, ensure_ascii=False, indent=1) + '\n')


def fix_markdown(path, fix, found):
    raw = read(path)
    nl = '\r\n' if '\r\n' in raw else '\n'
    lines = raw.split(nl)
    hits = find(path, 'file', lines)
    for j in hits:
        found.append((path, 'line %d' % (j + 1), lines[j - 1].strip()[:60],
                      lines[j].strip()))
    if fix and hits:
        for j in reversed(hits):
            lines.insert(j, '')
        write(path, nl.join(lines))


def main():
    fix = '--fix' in sys.argv
    found = []
    for path in sorted(targets()):
        if path.endswith('.ipynb'):
            fix_notebook(path, fix, found)
        else:
            fix_markdown(path, fix, found)
    for path, where, prev, rule in found:
        print('%s [%s]: %r directly after %r' % (path, where, rule, prev))
    print('%d setext-heading hazard(s) %s.' % (len(found),
                                              'fixed' if fix else 'found'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
