# Production
These are instructions on how to generate the slides and book. As a student, you don't need this. You can simply use the pre-compiled materials.

## Generating the online book
The book is built with **Jupyter Book 2** (MyST engine).

The old Jupyter Book v1 setup (`_config.yml` + `notebooks/_toc.yml`) is archived
in `_v1_backup/`, and the last v1 state is tagged `jupyter-book-v1` on GitHub
(use `git checkout jupyter-book-v1` to go back to it).

Install v2:

```
pip install jupyter-book        # v2.x — wraps the MyST engine
# or: npm install -g mystmd
```

On this workstation, `jupyter-book` 2.1.7 lives in the `pytorch` env:
`/data/python-envs/pytorch/bin/jupyter-book`.

All configuration now lives in a single `myst.yml` at the repository root:
metadata (title, author, copyright, logo, favicon), the bibliography,
exports, and the table of contents. The cover page is still `notebooks/index.md`.

Build the static site from the repository root (not from `notebooks/`):

```
jupyter book build --html
```

Gotcha: if your shell exports `PORT` (this one has `PORT=0`), unset it for the
build — MyST reads `PORT` and the static export then tries to fetch pages from
`http://localhost:0` and fails:

```
env -u PORT jupyter book build --html
```

The site is written to `_build/html/`.

### Serving the book locally

Option A — live-reload dev server (rebuilds content as you edit):

```
env -u PORT jupyter book start     # http://localhost:3000
```

Option B — serve the already-built static site on any port:

```
cd _build/html
python3 -m http.server 8811 --bind 127.0.0.1
```

(The site is a JS-driven static site: always open it through a web server,
not by double-clicking `index.html`.)

### Stopping local servers

If a server runs in a terminal, stop it with `Ctrl-C` there. Background
servers can be found and killed by port:

```
ss -tlnp | grep -E ':(3000|3001|8811)'   # find listeners + PIDs
kill <PID>                               # graceful
kill -9 <PID>                            # if it ignores SIGTERM
```

One-liners:

```
fuser -k 3000/tcp              # free port 3000
kill $(lsof -t -i:8811)        # free port 8811 (needs lsof)
```

`jupyter book start` spawns a Node "book-theme" server; if one is left over:

```
pkill -f "book-theme/server.js"
pkill -f "myst"
```

### Pushing the rendered book to GitHub Pages

For a project site (https://fum-cs.github.io/machine-learning/), rebuild with
the correct base URL first, then push `_build/html`:

```
BASE_URL=/machine-learning/ env -u PORT jupyter book build --html
pip install ghp-import
ghp-import -n -p -f ./_build/html
```


## LaTeX and PDF output

The book can also be rendered as LaTeX source and as a PDF. Both come out of the
**same pipeline**, so they are two stopping points rather than two pipelines:

```
notebooks/*.md, *.ipynb  ->  LaTeX (.tex)  ->  TeX engine (xelatex)  ->  PDF
                              ^-- build --tex      ^-- build --pdf
```

`build --tex` stops at the generated `.tex` (a deliverable you can edit, compile or
hand to a journal); `build --pdf` additionally compiles it. Notebook **code and
stored outputs are included** in both (the PDF is ~33 MB / ~400 pages), and
interactive HTML/widget output never makes it into a PDF.

This is separate from the per-lecture **slide** PDFs further down, which are
printed from the reveal.js HTML with `?print-pdf`.

### Prerequisites

```bash
pandoc --version          # any modern pandoc
xelatex --version         # a full TeX Live installation (the template uses xelatex)
```

On this workstation: pandoc 3.10 and TeX Live 2024.

### Configuring the exports

Both formats are configured in `project.exports` in `myst.yml`, using the
`plain_latex_book` template:

```yaml
  exports:
    - format: pdf
      template: plain_latex_book
      output: _build/exports/machine-learning-book.pdf
      articles: &book_articles      # the list is shared with the tex export
        - file: "notebooks/index.md"
          title: "Machine Learning"
          level: 0                   # 0 -> \chapter, 1 -> \section
        ...
    - format: tex
      template: plain_latex_book
      output: _build/exports/machine-learning-book-tex.zip
      articles: *book_articles
```

Three things matter here:

* **`level` per article.** By default MyST renders every page at `\section`,
  which is wrong for the `book` class. `0` makes a chapter (`\chapter`), `1` a
  section. The list above mirrors `project.toc`: the first page of each group is
  a chapter, the rest are sections (11 chapters, 24 sections).
* **`title` per article.** Without it, pages whose first heading is not an `H1`
  silently lose their heading and the chapter disappears from the book.
* The YAML anchor (`&book_articles` / `*book_articles`) avoids repeating 35
  entries twice.

The HTML site is *not* an export: `exports:` rejects `format: html`, so the site
stays in the top-level `site:` block and is built with `jupyter book build --html`.

### Building

```bash
env -u PORT jupyter book build --tex    # -> _build/exports/machine-learning-book-tex.zip
env -u PORT jupyter book build --pdf    # -> _build/exports/machine-learning-book.pdf  (~3 min)
env -u PORT jupyter book build -a       # every configured export
```

`PORT` must be unset here too (see the gotcha above).

### Reading the build log

The build runs `latexmk -f -xelatex -file-line-error -interaction=batchmode`, and
`-file-line-error` rewrites LaTeX's `!`-prefixed errors into `./file.tex:NN:` lines.
So **grepping the log for `^!` finds nothing even when the build is broken** — use:

```bash
grep -nE "^\./[^:]*:[0-9]+:" _build/temp/*/machine-learning-book.log
```

A hit means a real LaTeX error. An empty result *plus* MyST printing
`⛔️ LaTeX reported an error` means latexmk exited non-zero for another reason
(usually a missing file). Also useful:

```bash
grep -c "Missing character" _build/temp/*/machine-learning-book.log   # dropped glyphs
pdfinfo _build/exports/machine-learning-book.pdf                       # page count
```

### LaTeX gotchas in this repository

These are real failure modes found while producing the PDF; all of them are
silent in HTML and only bite in the LaTeX/PDF export:

1. **`---` directly after a non-blank line is a setext heading, not a rule.**
   Most cells here end with `---` as a visual separator. With no blank line
   before it, CommonMark reads it as a level-2 heading *titled by the previous
   line* — in `Naive-Bayes.ipynb` that put an `\includegraphics` inside a
   sectioning argument and xelatex died with
   `Paragraph ended before \@sect was complete`, i.e. exit code 1 and a log with
   no `^!` line. Fix: leave a blank line before the `---`.
   `tools/pdf_fix_setext_rules.py --fix` repairs all of them.
2. **Unicode arrows in prose.** myst-to-tex rewrites `→` to a bare `\rightarrow`
   *outside* math mode, and LaTeX then cascades into
   `Command \item invalid in math mode`. Write `$\rightarrow$` instead.
3. **`%` at the end of inline math.** In `$1 - (0.99)^2 \simeq 2%$` the `%`
   comments out the closing `$`, the math never closes, and xelatex exits 1
   while printing no error at all. Escape it: `2\%$`.
4. **Unicode glyphs the font lacks are dropped silently** (LaTeX only warns
   "Missing character"): `₁₂₃` subscripts, `ő`, `…`, `’`, box-drawing characters
   in stored outputs. Keep the prose ASCII or use math mode.
5. **BibTeX and non-ASCII fields.** Persian author names in `references.bib`
   make BibTeX fail with `name has a comma at the end`; wrap such fields in an
   extra brace group, or translate the entry. Note that BibTeX ignores `%`
   comments *inside* an entry. The remaining Persian in this file is confined to
   the `note` and `authorfa` fields, which BibTeX ignores, so it is harmless.
6. **`{bibliography}` in a page is not converted** and MyST logs
   `⛔️ Unhandled LaTeX conversion for node of "bibliography"`
   (`notebooks/index.md`). The master document still gets
   `\bibliographystyle{abbrvnat}` and the reference list, so the citations and
   the printed bibliography are correct; only the directive is dropped.
7. HTML is unaffected by all of the above, but fixing it in the source fixes
   both outputs.

### Helper scripts

`tools/` holds the diagnostics used for this work:

| script | purpose |
| --- | --- |
| `pdf_scan_unicode.py` | lists Unicode math symbols in prose (breaks the PDF) |
| `pdf_fix_prose_symbols.py` | wraps them in math mode |
| `pdf_fix_setext_rules.py` | finds/fixes `---` lines that become setext headings |
| `pdf_bisect_binary.py` | find the chapter that makes xelatex fail |
| `pdf_bisect_latex.py` | narrow a failing chapter down to a line |

`pdf_bisect_binary.py` takes the build directory and the master file:

```bash
python tools/pdf_bisect_binary.py _build/temp/mystXXXX machine-learning-book.tex
```

### Not covered by CI

`.github/workflows/deploy.yml` builds **HTML only**. The PDF is produced locally;
adding it to CI would need a job that installs TeX Live and uploads
`_build/exports/`.

## Generating slides
### Interactive slides
To generate interactive slides (as shown in the videos and lectures), you'll need to install the notebook extensions `rise` and `hide_input_all`.

```
conda install -c conda-forge rise
conda install -c conda-forge jupyter_nbextensions_configurator
jupyter nbextension enable rise
jupyter nbextension enable hide_input_all
```

You'll need to launch `jupyter notebook` since rise is not yet supported in jupyterlab. You'll see two new icons when you open a notebook. One will start the slideshow, and the other will hide the code. You'll need to set `interactive = True` in the first cell and then run all cells before starting the slideshow to allow the interactions.

### Static slides
You can generate slides from the notebooks using `nbconvert`. First, set `interactive = False` in the first cell of the notebook and rerun the notebooks to generate static versions of the interactive visualizations.

```
jupyter nbconvert --to slides --template reveal --SlidesExporter.reveal_theme=simple --no-input --post serve <NotebookName>
```

To generate PDF handouts, remove `#/` from the url and add `?print-pdf`, then print as PDF. 

Sidenote: Some PDF readers (e.g. Preview) sometimes show grey boxes around code examples, others (e.g. Acrobat Reader, Chrome) do not. Must be an artifact of some PDF readers.

#### Customization
I used a few tweaks to the slide theme by adding these to the css in `custom_reveal.css` in the `nbconvert` reveal template. Copy the styles from slides_html/custom.css to this file.

There also seems to be a bug in reveal that outputs text for hidden interactive elements. I added the following to `index.html.j2` of the same template:

```
<script>
var els = document.getElementsByTagName("pre");
for (var i = 0; i < els.length; ++i) {
    var el = els[i];
    if (el.textContent.indexOf("interactive") > -1) {
      el.style.display = "none";
    }
}
</script>
```

### Debug

for this error `jinja2.exceptions.UndefinedError: 'dict object' has no attribute 'image_relative'`, in tfcpu env in HP, I ran the programs in tfcpu, then build in pth with this option: `execute_notebooks: off`

(This was a Jupyter Book v1 workaround; in v2 notebooks are not executed by
default — use `jupyter book build --html --execute` to run them.)
