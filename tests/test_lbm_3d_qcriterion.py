"""Tests for src/lbm_3d_qcriterion.py -- Q-criterion field + isosurface.

Q = (1/2) (|Ω|² − |S|²). The textbook properties pinned here:
  1. Solid-body rotation (vorticity only): Q = ω² > 0, uniform in space.
  2. Pure simple shear (∂uₓ/∂y constant, all else 0): Q = 0 -- the
     strain and vorticity magnitudes are equal, they cancel.
  3. Quiescent flow: Q = 0.
  4. Body cell masking: ``compute_q_field(..., body=mask)`` zeros Q
     inside the mask.
  5. Marching-cubes round-trip: given a Q field with a known
     positive-Q region, ``extract_q_isosurface`` returns vertices and
     faces; ``None`` is returned when the level is outside the Q
     range.

Property #2 is the **diagnostic value** of Q-criterion: a naive
|Ω| threshold would identify uniform shear (an airfoil boundary
layer, say) as a vortex. Q correctly disqualifies it.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.lbm_3d_qcriterion import (
    compute_q_field,
    extract_q_isosurface,
    suggest_q_level,
)

# ---------------------------------------------------------------------------
# Analytic test fields
# ---------------------------------------------------------------------------


class TestComputeQField:
    def test_solid_body_rotation_gives_positive_uniform_Q(self):
        """u = ω × r, with ω = (0, 0, ωz). Strain is zero, the only
        non-zero gradient terms are ∂uₓ/∂y = -ωz and ∂u_y/∂x = +ωz.
        W_xy = -ωz, so |Ω|² = 2·ωz². Q = ωz² everywhere.

        This is THE canonical Q > 0 case -- pure rotation, no strain.
        """
        N = 16
        omega = 0.05
        ys = np.arange(N, dtype=np.float32)[None, :, None] - N // 2
        xs = np.arange(N, dtype=np.float32)[:, None, None] - N // 2
        ux = np.broadcast_to(-omega * ys, (N, N, N)).astype(np.float32).copy()
        uy = np.broadcast_to(omega * xs, (N, N, N)).astype(np.float32).copy()
        uz = np.zeros((N, N, N), dtype=np.float32)
        Q = compute_q_field(ux, uy, uz)
        # Interior cells should be ~ ωz². Tolerance is loose because
        # np.gradient uses one-sided differences at the boundary which
        # we exclude. ωz = 0.05 → Q = 0.0025.
        Q_interior = Q[2:-2, 2:-2, 2:-2]
        expected = omega * omega
        assert np.allclose(Q_interior, expected, atol=1e-6), (
            f"Q in solid-body rotation should be uniform ~{expected:.4g}, "
            f"got range [{Q_interior.min():.4g}, {Q_interior.max():.4g}]"
        )
        assert (Q_interior > 0).all(), "Q must be strictly positive in pure rotation"

    def test_pure_shear_gives_zero_Q(self):
        """uₓ = γ·y, u_y = u_z = 0. Simple shear has equal-magnitude
        strain and vorticity:
            S_xy = γ/2,  Ω_xy = γ/2  → both contribute (γ/2)² × 2
            |S|² = γ²/2,  |Ω|² = γ²/2
            Q = (1/2)(γ²/2 − γ²/2) = 0.

        This is what makes Q-criterion better than |Ω| -- a uniform
        boundary layer is correctly identified as NOT a vortex.
        """
        N = 16
        gamma = 0.05
        ys = np.arange(N, dtype=np.float32)[None, :, None]
        ux = np.broadcast_to(gamma * ys, (N, N, N)).astype(np.float32).copy()
        uy = np.zeros((N, N, N), dtype=np.float32)
        uz = np.zeros((N, N, N), dtype=np.float32)
        Q = compute_q_field(ux, uy, uz)
        # Interior should be exactly zero (modulo float32 round-off).
        Q_interior = Q[2:-2, 2:-2, 2:-2]
        assert np.allclose(Q_interior, 0.0, atol=1e-10), (
            f"Q in pure shear should be zero (strain == vorticity), "
            f"got max abs {np.abs(Q_interior).max():.4g}"
        )

    def test_quiescent_flow_gives_zero_Q(self):
        """u = 0 everywhere → all gradients zero → Q = 0."""
        N = 8
        u = np.zeros((N, N, N), dtype=np.float32)
        Q = compute_q_field(u, u, u)
        assert np.abs(Q).max() == 0.0

    def test_strain_only_field_gives_negative_Q(self):
        """Plane extensional strain: uₓ = ε·x, u_y = -ε·y, u_z = 0.
        Divergence-free, irrotational. |Ω|² = 0, |S|² = 2·ε².
        Q = -ε² < 0. Pinning the Q < 0 sign in strain-dominated flow.
        """
        N = 12
        eps = 0.03
        xs = np.arange(N, dtype=np.float32)[:, None, None] - N // 2
        ys = np.arange(N, dtype=np.float32)[None, :, None] - N // 2
        ux = np.broadcast_to(eps * xs, (N, N, N)).astype(np.float32).copy()
        uy = np.broadcast_to(-eps * ys, (N, N, N)).astype(np.float32).copy()
        uz = np.zeros((N, N, N), dtype=np.float32)
        Q = compute_q_field(ux, uy, uz)
        Q_interior = Q[2:-2, 2:-2, 2:-2]
        # |S|² = S_xx² + S_yy² = ε² + ε² = 2 ε². |Ω|² = 0.
        # Q = (1/2) (0 - 2 ε²) = -ε².
        expected = -eps * eps
        assert np.allclose(Q_interior, expected, atol=1e-6), (
            f"Q in extensional strain should be ~{expected:.4g}, got "
            f"range [{Q_interior.min():.4g}, {Q_interior.max():.4g}]"
        )

    def test_body_mask_zeros_interior(self):
        """A solid cell in the middle of an otherwise-rotating field
        should have Q = 0 -- the LBM driver pins u = 0 inside body
        cells, and the np.gradient produces spurious peaks at the
        surface discontinuity.
        """
        N = 12
        omega = 0.05
        ys = np.arange(N, dtype=np.float32)[None, :, None] - N // 2
        xs = np.arange(N, dtype=np.float32)[:, None, None] - N // 2
        ux = np.broadcast_to(-omega * ys, (N, N, N)).astype(np.float32).copy()
        uy = np.broadcast_to(omega * xs, (N, N, N)).astype(np.float32).copy()
        uz = np.zeros((N, N, N), dtype=np.float32)
        # Solid cube at the centre.
        body = np.zeros((N, N, N), dtype=bool)
        body[4:8, 4:8, 4:8] = True
        # Pin u to zero in the body, as the LBM driver does.
        ux = np.where(body, np.float32(0.0), ux)
        uy = np.where(body, np.float32(0.0), uy)
        Q = compute_q_field(ux, uy, uz, body=body)
        # Inside the body, Q must be zero (the mask zeroes it).
        assert (Q[body] == 0.0).all(), "Q must be zero inside body cells"


# ---------------------------------------------------------------------------
# Isosurface extraction
# ---------------------------------------------------------------------------


class TestExtractQIsosurface:
    def test_returns_none_when_level_above_max(self):
        """If level > Q.max(), there is no isosurface -- return None."""
        Q = np.linspace(-1.0, 1.0, 27, dtype=np.float32).reshape(3, 3, 3)
        result = extract_q_isosurface(Q, level=2.0)
        assert result is None

    def test_returns_none_when_level_below_min(self):
        Q = np.linspace(-1.0, 1.0, 27, dtype=np.float32).reshape(3, 3, 3)
        result = extract_q_isosurface(Q, level=-2.0)
        assert result is None

    def test_returns_vertices_and_faces_for_valid_level(self):
        """Build a Q field with a known positive bump and extract its
        isosurface. Verify the mesh has finite vertices and faces."""
        # Gaussian blob centred in the grid.
        N = 24
        xs, ys, zs = np.meshgrid(
            np.arange(N), np.arange(N), np.arange(N), indexing="ij"
        )
        Q = np.exp(
            -0.05 * ((xs - N / 2) ** 2 + (ys - N / 2) ** 2 + (zs - N / 2) ** 2)
        ).astype(np.float32)
        result = extract_q_isosurface(Q, level=0.5)
        assert result is not None
        verts, faces = result
        # Mesh sanity: at least one triangle, finite coordinates.
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert verts.shape[0] >= 3, f"need >=3 vertices, got {verts.shape[0]}"
        assert faces.shape[0] >= 1, f"need >=1 face, got {faces.shape[0]}"
        assert np.all(np.isfinite(verts))
        # Face indices must be valid into the vertex array.
        assert faces.max() < verts.shape[0]
        assert faces.min() >= 0


# ---------------------------------------------------------------------------
# Level suggestion helper
# ---------------------------------------------------------------------------


class TestSuggestQLevel:
    def test_returns_fraction_of_max_for_positive_field(self):
        Q = np.zeros((4, 4, 4), dtype=np.float32)
        Q[2, 2, 2] = 1.0
        level = suggest_q_level(Q, fraction=0.10)
        assert level == pytest.approx(0.1, abs=1e-6)

    def test_returns_tiny_positive_for_no_positive_Q(self):
        """If Q is everywhere <= 0 (no vortex regions), suggest_q_level
        returns a tiny positive number so the downstream isosurface
        call returns None instead of carving a surface inside the
        strain region.
        """
        Q = -np.ones((4, 4, 4), dtype=np.float32)
        level = suggest_q_level(Q)
        assert level > 0.0
        assert level < 1.0
