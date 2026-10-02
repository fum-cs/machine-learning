"""Scan MyST content for Unicode math symbols in prose (outside math mode).

These are converted to bare LaTeX macros (e.g. U+2192 -> \\rightarrow) by
myst-to-tex, which produces invalid LaTeX when they are not inside math mode.
"""
import io
import json
import re
import subprocess
import sys
import collections

FENCE = chr(96) * 3
MATH_RE = re.compile(r'[$][$].*?[$][$]|[$][^$]*[$]')
INLINE_CODE = chr(96) + '[^' + chr(96) + ']*' + chr(96)

SYMBOLS = {
    '→': 'rightarrow', '←': 'leftarrow', '⇒': 'Rightarrow',
    '⇔': 'Leftrightarrow', '↔': 'leftrightarrow', '↦': 'mapsto',
    '×': 'times', '÷': 'div', '≤': 'leq', '≥': 'geq',
    '≠': 'neq', '≈': 'approx', '≡': 'equiv', '∞': 'infty',
    '√': 'sqrt', '∈': 'in', '∉': 'notin', '⊂': 'subset',
    '∑': 'sum', '∏': 'prod', '∫': 'int', '∇': 'nabla',
    '∂': 'partial', '⋯': 'ldots', '…': 'ldots', '°': 'circ',
    '−': '-', '±': 'pm', '→': 'rightarrow',
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'epsilon', 'θ': 'theta', 'λ': 'lambda', 'μ': 'mu',
    'ν': 'nu', 'π': 'pi', 'ρ': 'rho', 'σ': 'sigma',
    'τ': 'tau', 'φ': 'phi', 'χ': 'chi', 'ω': 'omega',
    'Δ': 'Delta', 'Σ': 'Sigma', 'Ω': 'Omega', 'Φ': 'Phi',
}


def prose_lines(text):
    """Yield (line_number, line) for prose only (not code fences / math / code spans)."""
    in_fence = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(FENCE):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        yield i, MATH_RE.sub('', re.sub(INLINE_CODE, '', line))


def main():
    files = subprocess.run(['git', 'ls-files'], capture_output=True,
                           text=True).stdout.splitlines()
    counts = collections.Counter()
    where = collections.defaultdict(list)
    for f in files:
        if f == 'README.md' or not (f.endswith('.md') or f.endswith('.ipynb')):
            continue
        if f.endswith('.ipynb'):
            nb = json.load(io.open(f, encoding='utf-8'))
            chunks = []
            for idx, cell in enumerate(nb.get('cells', [])):
                if cell.get('cell_type') == 'markdown':
                    chunks.append(('cell%d' % idx, ''.join(cell.get('source', []))))
            texts = chunks
        else:
            texts = [('', io.open(f, encoding='utf-8', errors='replace').read())]
        for tag, text in texts:
            for lineno, line in prose_lines(text):
                for ch, latex in SYMBOLS.items():
                    if ch in line:
                        counts[ch] += line.count(ch)
                        where[ch].append('%s%s:%d' % (f, tag, lineno))
    total = 0
    for ch, n in counts.most_common():
        total += n
        locs = where[ch]
        print('%s  %-10s x%-3d  %d location(s)' % (ch, SYMBOLS[ch], n, len(locs)))
        for loc in locs[:4]:
            print('        ', loc)
        if len(locs) > 4:
            print('         ... +%d more' % (len(locs) - 4))
    print('\nTOTAL unicode-math symbols in prose:', total)


if __name__ == '__main__':
    sys.exit(main())