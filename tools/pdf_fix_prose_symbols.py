"""Make MyST prose safe for the LaTeX/PDF export (generic version).

Two failure modes are invisible in HTML but break the PDF build:

1. Unicode arrows in prose. myst-to-tex rewrites U+2192 to a bare `\\rightarrow`
   outside math mode, which LaTeX rejects and cascades into
   "Command \\item invalid in math mode". Wrapping them as `$\\rightarrow$`
   renders correctly in both HTML (MathJax) and LaTeX.
2. A percent sign at the end of inline math. In `$a \\simeq 2%$` the `%`
   comments out the closing `$`, the math never terminates, and xelatex exits
   non-zero while printing no error at all. Escaping it as `\\%` fixes it.

Scans every tracked Markdown/notebook file, touching only prose (never code
fences, inline code or existing math). Idempotent: already-correct text is left
alone.
"""
import io
import json
import re
import subprocess
import sys

FENCE = chr(96) * 3
MATH_SPAN = re.compile(r'[$][$].*?[$][$]|[$][^$]*[$]')
INLINE_CODE = chr(96) + '[^' + chr(96) + ']*' + chr(96)
PROTECT = re.compile(r'(?s)(' + FENCE + r'.*?' + FENCE + r'|'
                     + MATH_SPAN.pattern + r'|' + INLINE_CODE + r')')
BS = chr(92)

ARROWS = {
    '\u2192': '$' + BS + 'rightarrow$',   # ->
    '\u21d2': '$' + BS + 'Rightarrow$',   # =>
}
# '%' directly before the closing '$' of inline math, unless already escaped
PERCENT_IN_MATH = re.compile(r'(?<!' + re.escape(BS) + r')%(?=[$])')


def fix_text(text):
    """Return (new_text, n_arrows, n_percents)."""
    arrows = 0
    out = []
    for line in text.splitlines(keepends=True):
        parts = PROTECT.split(line)
        for i in range(0, len(parts), 2):          # even indices = prose
            for ch, rep in ARROWS.items():
                if ch in parts[i]:
                    arrows += parts[i].count(ch)
                    parts[i] = parts[i].replace(ch, rep)
        out.append(''.join(parts))
    new = ''.join(out)

    percents = 0
    fixed = []
    for line in new.splitlines(keepends=True):
        if '%' in line:
            parts = re.split('(' + MATH_SPAN.pattern + ')', line)
            for i in range(1, len(parts), 2):     # odd indices = math spans
                newmath, n = PERCENT_IN_MATH.subn(BS + '%', parts[i])
                percents += n
                parts[i] = newmath
            line = ''.join(parts)
        fixed.append(line)
    return ''.join(fixed), arrows, percents


def fix_markdown(path):
    raw = io.open(path, 'rb').read()
    crlf = b'\r\n' in raw
    text = raw.decode('utf-8').replace('\r\n', '\n')
    new, a, p = fix_text(text)
    if a or p:
        out = new.replace('\n', '\r\n') if crlf else new
        io.open(path, 'wb').write(out.encode('utf-8'))
    return a, p


def fix_notebook(path):
    raw = io.open(path, 'rb').read()
    crlf = b'\r\n' in raw
    nb = json.loads(raw.decode('utf-8').replace('\r\n', '\n'))
    a = p = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'markdown':
            continue
        src = cell.get('source')
        if isinstance(src, list):
            joined = ''.join(src)
        elif isinstance(src, str):
            joined = src
        else:
            continue
        new, ca, cp = fix_text(joined)
        a += ca
        p += cp
        if ca or cp:
            if isinstance(src, list):
                cell['source'] = new.splitlines(keepends=True)
            else:
                cell['source'] = new
    if a or p:
        out = json.dumps(nb, indent=1, ensure_ascii=False)
        out = out.replace('\n', '\r\n') + '\r\n' if crlf else out + '\n'
        io.open(path, 'wb').write(out.encode('utf-8'))
        json.loads(io.open(path, encoding='utf-8').read().replace('\r\n', '\n'))
    return a, p


def main():
    files = subprocess.run(['git', 'ls-files'], capture_output=True,
                           text=True).stdout.splitlines()
    total_a = total_p = 0
    for f in files:
        if f == 'README.md' or not f.endswith(('.md', '.ipynb')):
            continue
        a, p = fix_notebook(f) if f.endswith('.ipynb') else fix_markdown(f)
        if a or p:
            print('%-62s %d arrow(s), %d percent(s)' % (f, a, p))
            total_a += a
            total_p += p
    print('total: %d arrow(s), %d percent(s)' % (total_a, total_p))
    return 0


if __name__ == '__main__':
    sys.exit(main())