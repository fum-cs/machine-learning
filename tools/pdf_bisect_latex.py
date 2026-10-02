"""Bisect which included chapter makes xelatex return a non-zero exit code.

Compiles the generated book with \\includeonly set to one article at a time and
reports the exit status plus the number of `^!` errors in the log.
"""
import io
import os
import subprocess
import sys

BUILD = sys.argv[1] if len(sys.argv) > 1 else '.'
BS = chr(92)


def main():
    os.chdir(BUILD)
    master = io.open('mfds-book.tex', encoding='utf-8').read()
    includes = []
    i = 0
    while True:
        i = master.find(BS + 'include{', i)
        if i < 0:
            break
        j = master.find('}', i)
        includes.append(master[i + len(BS) + 8:j])
        i = j
    print('includes:', len(includes))
    targets = [n for n in includes if any(k in n for k in (
        'clustering-validation', 'image-clustering', 'pca-part1', 'norms',
        'eigendecomposition', 'svd-image-compression', 'compressed-sensing',
        'course-opening', 'course-synthesis', 'low-rank'))]
    print('testing %d candidate(s)' % len(targets))
    for name in targets:
        patched = master.replace(
            BS + 'begin{document}',
            BS + 'includeonly{' + name + '}' + chr(10) + BS + 'begin{document}', 1)
        io.open('bisect.tex', 'w', encoding='utf-8').write(patched)
        r = subprocess.run(['xelatex', '-interaction=batchmode',
                            '-file-line-error', 'bisect.tex'],
                           capture_output=True, text=True)
        errs = subprocess.run(['grep', '-c', '^!', 'bisect.log'],
                              capture_output=True, text=True).stdout.strip()
        print('%-52s exit=%s errors=%s' % (name[:50], r.returncode, errs))


if __name__ == '__main__':
    main()