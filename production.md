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
