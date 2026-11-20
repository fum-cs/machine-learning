"""Replace plotly HTML outputs in a notebook with static PNG images.

MyST/Jupyter Book stores notebook ``text/html`` outputs in its page JSON and
re-creates them in the browser, where the plotly ``require(["plotly"], ...)``
bootstrap never runs and no plotly.js is loaded: the figure is silently blank
on any static host. Plotly's own mime type is not rendered either, so the only
representation that survives a static build is an image. This script renders
each stored figure with kaleido and rewrites the output as ``image/png``.

Usage: python tools/plotly_to_png.py notebooks/Some-Notebook.ipynb [...]
"""

import base64
import io
import json
import sys
from json import JSONDecoder

import plotly.graph_objects as go

DEC = JSONDecoder()


def _skip(text, i, chars=" \t\r\n,"):
    while i < len(text) and text[i] in chars:
        i += 1
    return i


def figure_from_html(html):
    """Return the go.Figure of a plotly notebook output, or None."""
    if "Plotly.newPlot(" not in html:
        return None
    i = html.find("Plotly.newPlot(") + len("Plotly.newPlot(")
    i = _skip(html, i)
    try:
        _, i = DEC.raw_decode(html, i)                       # the div id
        i = _skip(html, i)
        data, i = DEC.raw_decode(html, i)
        i = _skip(html, i)
        layout, _ = DEC.raw_decode(html, i)
    except ValueError:
        return None
    if not isinstance(data, list) or not isinstance(layout, dict):
        return None
    return go.Figure(data=data, layout=layout)


def png_b64(fig):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def convert(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    done = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code" or not cell.get("outputs"):
            continue
        new_outputs = []
        figures = 0
        for out in cell["outputs"]:
            html = out.get("data", {}).get("text/html")
            if html is None:
                new_outputs.append(out)
                continue
            html = html if isinstance(html, str) else "".join(html)
            fig = figure_from_html(html)
            if fig is None:
                continue            # plotly.js bootstrap / other html: unusable when static
            new_outputs.append({
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": png_b64(fig)},
            })
            figures += 1
        if figures:
            cell["outputs"] = new_outputs
            done += figures
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return done


def main(paths):
    for path in paths:
        print(f"{path}: {convert(path)} figure(s) converted to PNG")


if __name__ == "__main__":
    main(sys.argv[1:] or sys.exit(__doc__))
