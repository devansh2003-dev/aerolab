"""Custom Streamlit components shipped with AeroLab.

These are loaded via streamlit.components.v1.declare_component, which
serves the HTML/JS bundle in an iframe and bridges back to Python via
the official postMessage protocol. Each subpackage has a frontend/
folder with an index.html that uses the bridge.
"""
from .polygon_drawer import polygon_drawer

__all__ = ["polygon_drawer"]
