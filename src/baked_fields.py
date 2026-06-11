"""Pre-baked 3D flow field storage (D-10 in 3D_PHASE0_DECISIONS.md).

A baked field is a steady-state (or quasi-steady) snapshot of the LBM
solver output for one preset configuration, stored as a compressed
``.npz`` and shipped with the app so the consumer-mode Streamlit page
can replay the smoke-particle viz without recomputing the kernel on
every visit. The math is decided by what we cached; the page is just
playback.

D-10 locks two choices:

  * **float16 storage, float32 dequantise at runtime.** A 64 × 32 × 32
    grid × 4 channels (rho, ux, uy, uz) at float32 is 1 MB; at float16
    it's 0.5 MB. The float16 round-trip introduces at most 1e-3
    relative error on velocity values of order 0.05 -- well below the
    smoke-particle integration error and the eyeball threshold on
    Plotly Scatter3d. Smaller grids barely matter; the format change
    matters at 96³ and above.
  * **Cache key = canonical hash of the bake parameters,** NOT the raw
    bytes of any uploaded mesh. For built-in presets (sphere, etc.)
    the hash is over the parameter dict. For uploaded meshes the
    hash will be of the voxelised body mask -- D-9 territory, not
    yet wired up.

The schema is versioned (``meta["version"]``) so future format changes
can be detected and refused cleanly instead of silently misreading
half-compatible files.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class BakedField:
    """Loaded baked field. Float arrays are float32 (dequantised from
    on-disk float16); ``body`` is bool.

    ``meta`` is the manifest dict captured at bake time: preset name,
    grid, simulation parameters, body params, bake timestamp, and the
    canonical parameter hash.
    """
    rho: np.ndarray         # (Nx, Ny, Nz) float32
    ux: np.ndarray          # (Nx, Ny, Nz) float32
    uy: np.ndarray          # (Nx, Ny, Nz) float32
    uz: np.ndarray          # (Nx, Ny, Nz) float32
    body: np.ndarray        # (Nx, Ny, Nz) bool
    meta: dict[str, Any]

    @property
    def Nx(self) -> int:
        return int(self.ux.shape[0])

    @property
    def Ny(self) -> int:
        return int(self.ux.shape[1])

    @property
    def Nz(self) -> int:
        return int(self.ux.shape[2])

    @property
    def preset_name(self) -> str:
        return str(self.meta.get("preset_name", "unknown"))


def canonical_param_hash(params: dict[str, Any]) -> str:
    """SHA-256 over the JSON-serialized parameter dict with sorted keys.

    Stable across reorderings of the same dict but sensitive to any
    value change. Used as the cache key so a user re-baking the same
    preset produces a byte-identical hash, but bumping (say) ``n_steps``
    invalidates the cache.

    Keys whose value is a float should be passed in their original
    float form; ``json.dumps`` will serialise them deterministically.
    """
    # D-9: numpy scalars (np.float64, np.int64) are not directly JSON-
    # serialisable -- a baked field built with `cy = grid.shape[1] / 2`
    # (np.float64) used to raise TypeError here. `default=float` casts
    # any unknown numpy scalar to plain Python float. Integers also
    # round-trip cleanly through float for hashing.
    blob = json.dumps(
        params, sort_keys=True, separators=(",", ":"), default=float,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_baked_field(
    path: str | Path,
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    body: np.ndarray,
    *,
    preset_name: str,
    bake_params: dict[str, Any],
) -> dict[str, Any]:
    """Quantise to float16 and write a compressed .npz.

    Parameters
    ----------
    path
        Destination .npz file. Parent directory must exist.
    rho, ux, uy, uz
        (Nx, Ny, Nz) float arrays. Will be cast to float16 for storage.
    body
        (Nx, Ny, Nz) bool array. Stored as-is.
    preset_name
        Human-readable identifier (used in the loaded ``meta``).
    bake_params
        Dict capturing the simulation configuration: Nx, Ny, Nz, u_in,
        nu, n_steps, scheme, body type, body params, etc. Used to
        compute the canonical hash and stored in ``meta`` for later
        inspection / replay.

    Returns
    -------
    meta : dict
        The full manifest that was persisted (including the auto-
        generated ``version``, ``hash``, ``ts_baked``).
    """
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(
            f"Parent directory {path.parent} does not exist. Create it "
            f"before calling save_baked_field."
        )

    # Shape sanity. All four scalar fields and the body mask must agree.
    shape = ux.shape
    for name, arr in (("rho", rho), ("uy", uy), ("uz", uz), ("body", body)):
        if arr.shape != shape:
            raise ValueError(
                f"Shape mismatch: ux is {shape} but {name} is {arr.shape}"
            )

    rho16 = rho.astype(np.float16, copy=False)
    ux16 = ux.astype(np.float16, copy=False)
    uy16 = uy.astype(np.float16, copy=False)
    uz16 = uz.astype(np.float16, copy=False)
    body_b = body.astype(np.bool_, copy=False)

    meta = dict(bake_params)
    meta["version"] = SCHEMA_VERSION
    meta["preset_name"] = preset_name
    meta["hash"] = canonical_param_hash(bake_params)
    meta["ts_baked"] = datetime.now(UTC).isoformat()
    meta["shape"] = list(shape)
    meta["dtype_storage"] = "float16"
    meta["dtype_runtime"] = "float32"

    meta_json = json.dumps(meta, sort_keys=True)

    np.savez_compressed(
        path,
        rho=rho16,
        ux=ux16,
        uy=uy16,
        uz=uz16,
        body=body_b,
        meta=np.array(meta_json),
    )
    return meta


def load_baked_field(path: str | Path) -> BakedField:
    """Read a .npz produced by :func:`save_baked_field` and dequantise.

    Float arrays are returned as float32; the body mask is bool. The
    manifest is parsed from the embedded JSON. A ``ValueError`` is
    raised if the schema version is newer than this module knows about
    -- we'd rather refuse than silently misread.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"baked field not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        # Manifest first so we can validate before paying the load
        # cost for the arrays.
        if "meta" not in data:
            raise ValueError(
                f"{path} does not contain a `meta` entry -- not a "
                f"baked-field file or written by a different tool."
            )
        meta_json = str(data["meta"])
        meta = json.loads(meta_json)

        version = meta.get("version", 0)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema version {version}; this module "
                f"only handles version {SCHEMA_VERSION}. Bump or "
                f"migrate the file."
            )

        # Cross-check every required array key BEFORE the load so a
        # truncated / partially-written bake raises a friendly ValueError
        # instead of a KeyError leaking from inside the np.load block.
        _required = ("rho", "ux", "uy", "uz", "body")
        _present = set(data.files)
        _missing = [k for k in _required if k not in _present]
        if _missing:
            raise ValueError(
                f"{path} is missing required arrays {_missing!r}. The "
                f"file may be truncated or written by an older tool; "
                f"expected keys are {list(_required)!r}."
            )

        rho = np.asarray(data["rho"]).astype(np.float32, copy=False)
        ux = np.asarray(data["ux"]).astype(np.float32, copy=False)
        uy = np.asarray(data["uy"]).astype(np.float32, copy=False)
        uz = np.asarray(data["uz"]).astype(np.float32, copy=False)
        body = np.asarray(data["body"]).astype(np.bool_, copy=False)

    return BakedField(rho=rho, ux=ux, uy=uy, uz=uz, body=body, meta=meta)


def list_baked_fields(directory: str | Path) -> list[Path]:
    """Return all .npz files in ``directory`` that look like baked
    fields (by extension only -- the schema validation happens at
    load time). Sorted alphabetically for stable UI ordering.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.npz"))


__all__ = [
    "SCHEMA_VERSION",
    "BakedField",
    "canonical_param_hash",
    "save_baked_field",
    "load_baked_field",
    "list_baked_fields",
]
