"""Binary-search which set of included chapters makes xelatex exit non-zero.

xelatex can return a non-zero status with a completely clean log (no `^!` line),
which makes latexmk - and therefore the MyST PDF export - report a failure. This
compiles the generated book with `\\includeonly` restricted to subsets of the
included articles to find the offending chapter.

Usage: pdf_bisect_binary.py <build-dir> [master.tex]
"""
import io
import os
import subprocess
import sys

BUILD = sys.argv[1] if len(sys.argv) > 1 else '.'
MASTER = sys.argv[2] if len(sys.argv) > 2 else None
BS = chr(92)


def main():
    os.chdir(BUILD)
    master = MASTER or [f for f in os.listdir('.')
                        if f.endswith('.tex') and '-' + BS + 'include{' not in f][0]
    text = io.open(master, encoding='utf-8').read()
    includes = []
    i = 0
    while True:
        i = text.find(BS + 'include{', i)
        if i < 0:
            break
        j = text.find('}', i)
        includes.append(text[i + len(BS) + 8:j])
        i = j
    print('master:', master, '| includes:', len(includes))

    def compile_subset(names, tag):
        patched = text.replace(
            BS + 'begin{document}',
            BS + 'includeonly{' + ','.join(names) + '}' + chr(10)
            + BS + 'begin{document}', 1)
        name = 'bisect-%s.tex' % tag
        io.open(name, 'w', encoding='utf-8').write(patched)
        r = subprocess.run(['xelatex', '-interaction=batchmode',
                            '-file-line-error', name], capture_output=True)
        return r.returncode

    print('baseline (all):', compile_subset(includes, 'all'))
    lo, hi, step = 0, len(includes), 0
    while hi - lo > 1:
        step += 1
        mid = (lo + hi) // 2
        a = compile_subset(includes[lo:mid], 'a%d' % step)
        b = compile_subset(includes[mid:hi], 'b%d' % step)
        print('step %d: [%d:%d] rc=%s | [%d:%d] rc=%s'
              % (step, lo, mid, a, mid, hi, b))
        if a != 0:
            hi = mid
        elif b != 0:
            lo = mid
        else:
            print('  -> failure needs the combination of both halves')
            break
    print('candidate:', includes[lo:hi])


if __name__ == '__main__':
    main()