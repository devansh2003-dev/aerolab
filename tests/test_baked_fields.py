"""Tests for src/baked_fields.py -- pre-baked field save/load round-trip.

The data layer of D-10 (consumer-mode gallery). These tests pin:
  1. Round-trip preserves shapes and dtypes (float16 storage, float32
     runtime).
  2. Float16 quantisation error is small enough not to affect the
     downstream smoke-particle viz.
  3. Manifest is captured + reloaded and matches the original
     bake_params.
  4. The canonical hash is deterministic and value-sensitive.
  5. Schema-version mismatch on load raises cleanly.
  6. Missing files and bad parents are caught with informative errors.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.baked_fields import (
    SCHEMA_VERSION,
    BakedField,
    canonical_param_hash,
    list_baked_fields,
    load_baked_field,
    save_baked_field,
)


def _make_dummy_field(Nx=16, Ny=8, Nz=8, u_in=0.05):
    """Return a small (rho, ux, uy, uz, body) tuple with non-trivial
    structure so float16 quantisation errors are detectable.
    """
    rng = np.random.default_rng(seed=42)
    rho = (1.0 + 0.05 * rng.standard_normal((Nx, Ny, Nz))).astype(np.float32)
    # Velocity with realistic ranges so float16 round-off is testable.
    ux = (u_in * rng.uniform(0.5, 1.5, size=(Nx, Ny, Nz))).astype(np.float32)
    uy = (0.005 * rng.standard_normal((Nx, Ny, Nz))).astype(np.float32)
    uz = (0.005 * rng.standard_normal((Nx, Ny, Nz))).astype(np.float32)
    body = np.zeros((Nx, Ny, Nz), dtype=bool)
    body[Nx // 4 : Nx // 4 + 3, Ny // 3 : Ny // 3 + 3, Nz // 3 : Nz // 3 + 3] = True
    return rho, ux, uy, uz, body


# ---------------------------------------------------------------------------
# Canonical hash
# ---------------------------------------------------------------------------


class TestCanonicalParamHash:
    def test_deterministic_for_same_input(self):
        params = {"a": 1, "b": 2.5, "c": "hello"}
        h1 = canonical_param_hash(params)
        h2 = canonical_param_hash(params)
        assert h1 == h2

    def test_stable_under_key_reordering(self):
        a = {"a": 1, "b": 2, "c": 3}
        b = {"c": 3, "b": 2, "a": 1}
        assert canonical_param_hash(a) == canonical_param_hash(b)

    def test_changes_when_value_changes(self):
        a = {"u_in": 0.05}
        b = {"u_in": 0.06}
        assert canonical_param_hash(a) != canonical_param_hash(b)

    def test_returns_sha256_hex_string(self):
        h = canonical_param_hash({"x": 1})
        assert isinstance(h, str)
        assert len(h) == 64
        # All hex characters.
        int(h, 16)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_shapes_and_dtypes_preserved(self, tmp_path):
        rho, ux, uy, uz, body = _make_dummy_field()
        path = tmp_path / "test.npz"
        save_baked_field(
            path, rho, ux, uy, uz, body,
            preset_name="test_preset",
            bake_params={"Nx": 16, "Ny": 8, "Nz": 8, "u_in": 0.05},
        )
        loaded = load_baked_field(path)
        assert isinstance(loaded, BakedField)
        assert loaded.rho.shape == rho.shape
        assert loaded.ux.shape == ux.shape
        assert loaded.uy.shape == uy.shape
        assert loaded.uz.shape == uz.shape
        assert loaded.body.shape == body.shape
        assert loaded.rho.dtype == np.float32
        assert loaded.ux.dtype == np.float32
        assert loaded.uy.dtype == np.float32
        assert loaded.uz.dtype == np.float32
        assert loaded.body.dtype == np.bool_

    def test_float16_round_trip_error_below_tolerance(self, tmp_path):
        """Float16 has ~3 significant decimal digits. For velocities of
        order 0.05 the absolute round-off should be < 1e-4. This is
        well below the smoke-particle integration error and orders of
        magnitude below the eyeball threshold on Plotly Scatter3d.
        """
        rho, ux, uy, uz, body = _make_dummy_field()
        path = tmp_path / "rt.npz"
        save_baked_field(
            path, rho, ux, uy, uz, body,
            preset_name="rt", bake_params={},
        )
        loaded = load_baked_field(path)
        # Max absolute error per field. With float16 having ~1e-3
        # relative precision and our values ~0.05-1.0, ~1e-3 absolute
        # is the right gate.
        assert float(np.max(np.abs(loaded.rho - rho))) < 1e-2, (
            "rho round-trip exceeded float16 precision budget"
        )
        assert float(np.max(np.abs(loaded.ux - ux))) < 1e-3, (
            "ux round-trip exceeded float16 precision budget"
        )
        assert float(np.max(np.abs(loaded.uy - uy))) < 1e-3
        assert float(np.max(np.abs(loaded.uz - uz))) < 1e-3
        # body is bool so exact equality is required.
        assert np.array_equal(loaded.body, body)

    def test_meta_includes_bake_params_and_auto_fields(self, tmp_path):
        rho, ux, uy, uz, body = _make_dummy_field()
        path = tmp_path / "m.npz"
        bake_params = {
            "Nx": 16, "Ny": 8, "Nz": 8,
            "u_in": 0.05, "nu": 0.02,
            "scheme": "trt",
        }
        save_baked_field(
            path, rho, ux, uy, uz, body,
            preset_name="m_preset", bake_params=bake_params,
        )
        loaded = load_baked_field(path)
        # All bake_params must round-trip.
        for k, v in bake_params.items():
            assert loaded.meta[k] == v, f"meta[{k}] = {loaded.meta[k]!r}, expected {v!r}"
        # Auto-injected fields.
        assert loaded.meta["version"] == SCHEMA_VERSION
        assert loaded.meta["preset_name"] == "m_preset"
        assert len(loaded.meta["hash"]) == 64
        assert "ts_baked" in loaded.meta
        assert loaded.meta["dtype_storage"] == "float16"
        assert loaded.meta["dtype_runtime"] == "float32"

    def test_hash_matches_canonical_of_bake_params(self, tmp_path):
        """Verify the meta hash is exactly the canonical hash of the
        user-provided bake_params (NOT the auto-extended meta).
        Sensitive to whether the save function passes the correct
        dict to canonical_param_hash.
        """
        rho, ux, uy, uz, body = _make_dummy_field()
        path = tmp_path / "h.npz"
        bake_params = {"u_in": 0.05, "n_steps": 800}
        save_baked_field(
            path, rho, ux, uy, uz, body,
            preset_name="h", bake_params=bake_params,
        )
        loaded = load_baked_field(path)
        assert loaded.meta["hash"] == canonical_param_hash(bake_params)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_save_rejects_nonexistent_parent_directory(self, tmp_path):
        rho, ux, uy, uz, body = _make_dummy_field()
        bad_path = tmp_path / "does" / "not" / "exist" / "x.npz"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            save_baked_field(
                bad_path, rho, ux, uy, uz, body,
                preset_name="x", bake_params={},
            )

    def test_save_rejects_shape_mismatch(self, tmp_path):
        rho, ux, uy, uz, body = _make_dummy_field()
        bad_uy = uy[:, :, :-1]  # smaller in z
        path = tmp_path / "bad.npz"
        with pytest.raises(ValueError, match="Shape mismatch"):
            save_baked_field(
                path, rho, ux, bad_uy, uz, body,
                preset_name="x", bake_params={},
            )

    def test_load_rejects_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="baked field not found"):
            load_baked_field(tmp_path / "does_not_exist.npz")

    def test_load_rejects_wrong_schema_version(self, tmp_path):
        """A file written with a future schema_version must raise on
        load, not silently misinterpret the arrays.
        """
        rho, ux, uy, uz, body = _make_dummy_field()
        path = tmp_path / "future.npz"
        # Hand-craft an .npz with version = 999.
        meta = {
            "version": 999, "preset_name": "future",
            "hash": "x" * 64, "ts_baked": "2099-01-01T00:00:00",
            "shape": list(ux.shape),
        }
        np.savez_compressed(
            path,
            rho=rho.astype(np.float16), ux=ux.astype(np.float16),
            uy=uy.astype(np.float16), uz=uz.astype(np.float16),
            body=body,
            meta=np.array(json.dumps(meta)),
        )
        with pytest.raises(ValueError, match="schema version 999"):
            load_baked_field(path)

    def test_load_rejects_file_without_meta(self, tmp_path):
        """A .npz from some other tool (no `meta` entry) must raise
        cleanly, not silently load garbage.
        """
        path = tmp_path / "stranger.npz"
        np.savez_compressed(path, foo=np.zeros(4))
        with pytest.raises(ValueError, match="not a baked-field file"):
            load_baked_field(path)


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------


class TestListBakedFields:
    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert list_baked_fields(tmp_path) == []

    def test_returns_only_npz_files_sorted(self, tmp_path):
        # Touch a few files.
        for name in ("c.npz", "a.npz", "b.npz", "ignore_me.txt"):
            (tmp_path / name).touch()
        result = list_baked_fields(tmp_path)
        names = [p.name for p in result]
        assert names == ["a.npz", "b.npz", "c.npz"]

    def test_nonexistent_directory_returns_empty_list(self, tmp_path):
        # Should not raise -- the consumer page may call this before
        # any bake has happened.
        result = list_baked_fields(tmp_path / "does_not_exist")
        assert result == []
