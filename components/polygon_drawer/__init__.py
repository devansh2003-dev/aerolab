"""Click-to-place polygon drawer custom component.

The streamlit-drawable-canvas polygon mode requires a double-click (or
right-click) to close the shape, which most first-time sketchers never
discover. This custom component implements the natural "click to place
vertices, click the green start dot to close" workflow that everyone
expects from drawing tools (Adobe Illustrator, Figma, Inkscape, GIMP).

The component is registered via the standard Streamlit component bridge
(streamlit.components.v1.declare_component) so it can return polygon
data back to Python. The frontend is a single static index.html with
inline vanilla JS -- no React build step, no npm install -- which keeps
the Cloud cold-start fast and the dependency surface zero.

Returns
-------
A dict with these keys, or ``None`` until the user has interacted:
    vertices : list of {"x": float, "y": float}
        Pixel coordinates in canvas-image space (origin top-left, y down).
    closed   : bool
        True once the user has clicked the start point to close the loop.
    width    : int
        Canvas width in pixels (matches the value passed in).
    height   : int
        Canvas height in pixels (matches the value passed in).
"""
from __future__ import annotations

import os

import streamlit.components.v1 as components

# Path to the frontend folder served as the component's iframe content.
# Streamlit's declare_component only actually registers the component (and
# exposes its iframe URL at /component/{module_name}.{name}/...) when run
# inside a ScriptRunContext -- the side effect of importing this module
# during a script run is what wires the URL handler in. app.py imports
# polygon_drawer at module top-level so the registration happens before
# any tab can try to render the iframe.
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component_func = components.declare_component(
    "aerolab_polygon_drawer", path=_FRONTEND_DIR,
)


def polygon_drawer(
    width: int = 400,
    height: int = 200,
    key: str | None = None,
) -> dict | None:
    """Render the click-to-place polygon canvas.

    Parameters
    ----------
    width, height
        Canvas pixel dimensions. Default 400x200 (2:1 aspect, comfortable
        for sketching but rescaled to the LBM channel downstream).
    key
        Streamlit widget key. Reuse the same key to persist drawings
        across reruns.

    Returns
    -------
    None until the user has interacted at least once. After that, a dict
    with keys ``vertices`` (list of {x, y}), ``closed`` (bool), ``width``,
    ``height``. See the module docstring for the full contract.
    """
    return _component_func(
        width=int(width), height=int(height), key=key, default=None,
    )
