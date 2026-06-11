"""STL -> voxel body mask, for the 3D mesh-upload pipeline (D-9).

The 3D consumer gallery + dev-bench accept user-uploaded geometry as
STL files. This module is the read + voxelize half of that pipeline;
the LBM kernel consumes the resulting bool mask exactly the same way
it consumes built-in sphere / box masks (halfway bounce-back on
``True`` cells, optional Bouzidi q-field for built-in analytic
shapes).

Three layers, smallest dependency cost first:

  ``read_stl(path)``          binary or ASCII STL -> ``(N, 3, 3)``
                              float32 triangle-vertex array. No new
                              runtime deps (binary STL is a trivial
                              80 + 4 + 50 N byte format; ASCII is
                              line-based text).

  ``voxelize_triangles(...)`` ray-cast voxelization. Casts rays in
                              the +x direction; cells with odd
                              triangle-intersection count are
                              interior. Vectorised per-triangle over
                              the affected (j, k) bounding-box slab.
                              For a typical 1000-triangle mesh on a
                              64x32x32 grid this runs in well under a
                              second.

  ``voxel_mask_for_lbm(...)`` high-level wrapper. Auto-centres the
                              mesh in the channel, scales so the
                              longest extent fits ``body_extent_cells``,
                              voxelises, then applies one round of
                              ``scipy.ndimage.binary_closing`` so
                              small shell-rasterisation holes in
                              non-perfectly-manifold meshes don't
                              produce hollow interiors.

The voxelization is the **classic odd-parity ray-cast** -- see e.g.
Patil & Ravi 2005 *J. Computing & Inf. Sci. Engg.* **5**, 219, or
Nooruddin & Turk 2003. The algorithm assumes the input mesh is a
closed manifold; the morphological close in the high-level wrapper
is a safety net for slightly-non-manifold STLs (single missing
triangle, T-junctions). Pathological meshes (large holes,
self-intersections) produce visibly wrong masks; voxel_mask_for_lbm
guards the two pathological extremes (empty mask, whole-domain mask)
with friendly ValueErrors and refuses to silently clip the body when
the scaled bounding-box exceeds the grid.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing

_BINARY_STL_HEADER_BYTES = 80
_BINARY_STL_COUNT_BYTES = 4
_BINARY_STL_TRI_BYTES = 50  # 12 floats * 4 bytes + 2-byte attribute


def _is_binary_stl(raw: bytes) -> bool:
    """Heuristic: binary STL files have an exact size of
    ``80 + 4 + 50 N`` bytes where N is the triangle count stored at
    bytes 80--83. ASCII STL files start with ``solid`` and are
    line-based text. Both formats can begin with the string "solid"
    (binary STLs sometimes have it in the header), so length is the
    only reliable check.
    """
    if len(raw) < _BINARY_STL_HEADER_BYTES + _BINARY_STL_COUNT_BYTES:
        return False
    n_tri = struct.unpack_from("<I", raw, _BINARY_STL_HEADER_BYTES)[0]
    expected = (
        _BINARY_STL_HEADER_BYTES
        + _BINARY_STL_COUNT_BYTES
        + _BINARY_STL_TRI_BYTES * n_tri
    )
    return len(raw) == expected


def _read_stl_binary(raw: bytes) -> np.ndarray:
    n_tri = struct.unpack_from("<I", raw, _BINARY_STL_HEADER_BYTES)[0]
    # 50 bytes per triangle: 3 floats normal + 9 floats vertices + 2-byte
    # attribute. dtype mirrors the on-disk layout exactly so np.frombuffer
    # can slice it in one shot.
    tri_dtype = np.dtype([
        ("normal", "<f4", 3),
        ("v0", "<f4", 3),
        ("v1", "<f4", 3),
        ("v2", "<f4", 3),
        ("attr", "<u2"),
    ])
    body = raw[_BINARY_STL_HEADER_BYTES + _BINARY_STL_COUNT_BYTES:]
    records = np.frombuffer(body, dtype=tri_dtype, count=n_tri)
    # Stack the three vertex columns into a single (N, 3, 3) array; the
    # normals are not needed for voxelisation (we infer orientation from
    # vertex order during ray-casting).
    return np.stack([records["v0"], records["v1"], records["v2"]], axis=1)


def _read_stl_ascii(text: str) -> np.ndarray:
    # ASCII STL grammar (ignoring whitespace + case):
    #   solid <name>
    #     facet normal nx ny nz
    #       outer loop
    #         vertex x y z
    #         vertex x y z
    #         vertex x y z
    #       endloop
    #     endfacet
    #     ...
    #   endsolid <name>
    # We need just the vertex triples; everything else is structural.
    verts: list[tuple[float, float, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.lower().startswith("vertex"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed ASCII STL vertex line: {raw!r}")
        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    # C-13 (b): a truncated binary STL that happens to decode as ASCII
    # (random bytes that don't include "vertex" lines) falls through to
    # here with verts == []. Without this check we'd silently return a
    # (0, 3, 3) array; downstream voxelisation then produces an empty
    # mask whose error message ("mesh fell outside the grid") sends the
    # caller chasing the wrong root cause.
    if len(verts) == 0:
        raise ValueError(
            "ASCII STL contained no `vertex` lines -- possibly a "
            "truncated binary STL, an empty file, or a malformed text "
            "STL."
        )
    if len(verts) % 3 != 0:
        raise ValueError(
            f"ASCII STL has {len(verts)} vertices, not divisible by 3 -- "
            f"truncated or malformed file."
        )
    arr = np.asarray(verts, dtype=np.float32).reshape(-1, 3, 3)
    return arr


def read_stl(path: str | Path) -> np.ndarray:
    """Read an STL file (binary or ASCII).

    Returns
    -------
    triangles : ndarray of shape ``(N, 3, 3)``, dtype float32
        ``triangles[i]`` is the i-th triangle; ``triangles[i, j]`` is
        the j-th vertex of that triangle as ``(x, y, z)``. Coordinates
        are in the STL's native units -- ``voxel_mask_for_lbm`` is
        responsible for scaling onto the LBM grid.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is neither valid binary nor valid ASCII STL.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STL not found: {path}")
    raw = path.read_bytes()
    if len(raw) == 0:
        raise ValueError(f"STL file is empty: {path}")
    if _is_binary_stl(raw):
        return _read_stl_binary(raw)
    # Fall back to ASCII. Some binary STLs that happen to be size-mismatched
    # would land here; the ASCII parser will raise cleanly if the text
    # isn't valid.
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not a valid binary STL (wrong file size) and "
            f"not valid ASCII either ({exc})."
        ) from exc
    return _read_stl_ascii(text)


def voxelize_triangles(
    triangles: np.ndarray,
    grid_shape: tuple[int, int, int],
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_size: float = 1.0,
) -> np.ndarray:
    """Odd-parity ray-cast voxelization, rays along +x.

    Parameters
    ----------
    triangles : ndarray of shape ``(N, 3, 3)``
        Triangle vertices in the same units as ``origin`` and
        ``voxel_size`` (i.e. world units, not grid indices).
    grid_shape : tuple
        ``(Nx, Ny, Nz)`` voxel grid extent.
    origin : tuple, optional
        World-coordinate corner of voxel ``(0, 0, 0)``. Voxel ``(i, j, k)``
        spans ``[origin + (i, j, k) * voxel_size, origin + (i+1, j+1,
        k+1) * voxel_size]`` and its centre is at ``origin + (i + 0.5,
        j + 0.5, k + 0.5) * voxel_size``.
    voxel_size : float, optional
        Uniform voxel edge length in world units.

    Returns
    -------
    mask : ndarray of shape ``(Nx, Ny, Nz)``, dtype bool
        ``True`` where the voxel centre is interior to the mesh.

    Notes
    -----
    The mesh is assumed to be a closed manifold. Triangles whose plane
    is parallel to the x-axis (normal perpendicular to x) contribute
    zero +x intersections to any column and are skipped automatically
    by the degenerate-determinant guard.
    """
    Nx, Ny, Nz = grid_shape
    ox, oy, oz = origin
    vs = float(voxel_size)
    if vs <= 0.0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    # Per-column intersection lists. Use one flat list of (col_index,
    # x_hit) and bucket-sort at the end -- simpler than a list-of-lists.
    col_idx_buf: list[int] = []
    x_hit_buf: list[float] = []

    # Cell-centre coordinates along (y, z) -- precomputed once.
    y_centres = oy + (np.arange(Ny) + 0.5) * vs
    z_centres = oz + (np.arange(Nz) + 0.5) * vs

    for tri in triangles:
        v0, v1, v2 = tri[0], tri[1], tri[2]
        # 2D edges in the (y, z) projection.
        e0_y = v1[1] - v0[1]
        e0_z = v1[2] - v0[2]
        e1_y = v2[1] - v0[1]
        e1_z = v2[2] - v0[2]
        det = e0_y * e1_z - e1_y * e0_z
        # Triangle's plane parallel to x-axis -> zero (y, z) projected
        # area -> contributes nothing to +x ray casts. Skip.
        if abs(det) < 1e-12:
            continue
        inv_det = 1.0 / det

        # Bounding box in voxel indices (j, k).
        ys = (v0[1], v1[1], v2[1])
        zs = (v0[2], v1[2], v2[2])
        j_min = max(0, int(np.floor((min(ys) - oy) / vs)))
        j_max = min(Ny - 1, int(np.ceil((max(ys) - oy) / vs)))
        k_min = max(0, int(np.floor((min(zs) - oz) / vs)))
        k_max = min(Nz - 1, int(np.ceil((max(zs) - oz) / vs)))
        if j_max < j_min or k_max < k_min:
            continue

        # Vectorise inside-triangle test over the (j, k) bbox slab.
        ys_slab = y_centres[j_min:j_max + 1]                 # (Lj,)
        zs_slab = z_centres[k_min:k_max + 1]                 # (Lk,)
        Y, Z = np.meshgrid(ys_slab, zs_slab, indexing="ij")   # (Lj, Lk)
        dy = Y - v0[1]
        dz = Z - v0[2]
        u = (dy * e1_z - dz * e1_y) * inv_det
        v = (e0_y * dz - e0_z * dy) * inv_det
        w = 1.0 - u - v
        # Half-open inclusion: ``u > 0`` not ``u >= 0`` so a ray that
        # grazes a shared edge between two triangles isn't counted
        # twice. Using a small epsilon tolerance to be friendly to
        # exact-axis-aligned meshes like the unit cube the test suite
        # builds. The convention isn't critical for closed manifolds:
        # any consistent edge-inclusion rule preserves parity.
        eps = 1e-9
        inside = (u >= -eps) & (v >= -eps) & (w >= -eps)
        if not inside.any():
            continue

        # Where inside, the x-coordinate of the ray-plane intersection
        # is the barycentric interpolation of the triangle's vertex
        # x-coords.
        x_hit = w * v0[0] + u * v1[0] + v * v2[0]
        # Indices of inside cells, in (j, k) order. Flatten to a 1D
        # column index for the column-intersection buffer.
        jj, kk = np.where(inside)
        col_indices = (j_min + jj) * Nz + (k_min + kk)
        x_hits = x_hit[jj, kk]
        col_idx_buf.extend(col_indices.tolist())
        x_hit_buf.extend(x_hits.tolist())

    # Bucket the intersections into per-column lists.
    intersections: list[list[float]] = [[] for _ in range(Ny * Nz)]
    for cidx, x in zip(col_idx_buf, x_hit_buf, strict=True):
        intersections[cidx].append(x)

    mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    x_centres = ox + (np.arange(Nx) + 0.5) * vs
    # Deduplication tolerance for shared-edge intersections: two
    # triangles meeting at an edge each report the same x_hit for any
    # ray that grazes the shared edge. Counting both would corrupt
    # parity. Tolerance is one part in 10**7 of the voxel size --
    # well above float32 noise on a 64-cell domain but well below any
    # meaningful surface separation a real STL would carry.
    dedup_tol = max(vs, 1.0) * 1e-7
    for col in range(Ny * Nz):
        xs = intersections[col]
        if not xs:
            continue
        xs_sorted = np.sort(np.asarray(xs, dtype=np.float64))
        if xs_sorted.size > 1:
            # Keep only the first of each run of near-coincident values.
            keep = np.empty(xs_sorted.size, dtype=bool)
            keep[0] = True
            keep[1:] = np.diff(xs_sorted) > dedup_tol
            xs_sorted = xs_sorted[keep]
        # For each cell centre x_c in this column, count how many
        # intersections lie BEFORE x_c. Odd count = inside.
        counts = np.searchsorted(xs_sorted, x_centres, side="right")
        j, k = divmod(col, Nz)
        mask[:, j, k] = (counts & 1).astype(bool)
    return mask


def voxel_mask_for_lbm(
    stl_path: str | Path,
    Nx: int,
    Ny: int,
    Nz: int,
    *,
    body_extent_cells: float = 8.0,
    padding_cells: tuple[float, float, float] = (16.0, 12.0, 12.0),
    close_iters: int = 1,
) -> np.ndarray:
    """High-level wrapper. STL on disk -> ``(Nx, Ny, Nz)`` bool mask
    ready to drop into ``run_channel_smoke_trt`` or
    ``run_channel_smoke``.

    The mesh is centred in the channel cross-section ``(y, z)`` and
    placed ``padding_cells[0]`` cells downstream of the inflow so the
    Guo NEEM / equilibrium inflow band has clear fluid in front of
    the body. The mesh is then scaled uniformly so its **longest
    axis** spans ``body_extent_cells`` voxels. Aspect ratio is
    preserved -- you get a 1x scaled version of the input mesh.

    Parameters
    ----------
    stl_path : str or Path
        Input mesh file. Binary or ASCII STL.
    Nx, Ny, Nz : int
        LBM grid extent in lattice units.
    body_extent_cells : float, optional
        Longest-axis extent of the body after scaling, in lattice
        cells. Default 8.0; the built-in sphere preset uses R = 4 so
        diameter = 8 cells -- this default matches.
    padding_cells : tuple of three floats, optional
        Inflow-side padding (x), and cross-section margins (y, z).
        The body is placed so its bounding box starts at
        ``x = padding_cells[0]`` lattice cells from the inflow plane.
    close_iters : int, optional
        Number of ``scipy.ndimage.binary_closing`` iterations. 1 is
        enough to fill single-voxel holes from slightly-non-manifold
        meshes; 0 disables the closing step (use for already-clean
        meshes when you need exact preservation of small concavities).

    Returns
    -------
    body_mask : ndarray of shape ``(Nx, Ny, Nz)``, dtype bool
        ``True`` cells are solid; pass directly to the LBM driver.

    Raises
    ------
    ValueError
        If the STL parses but voxelises to an empty or
        whole-domain-filling mask (mis-scaled mesh, or
        non-manifold beyond what morphological close can fix).
    """
    if body_extent_cells <= 0:
        raise ValueError(
            f"body_extent_cells must be positive, got {body_extent_cells}"
        )
    triangles = read_stl(stl_path)
    if triangles.shape[0] == 0:
        raise ValueError(f"STL {stl_path} contains zero triangles.")

    # Compute the mesh's native bounding box and centroid.
    flat = triangles.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    extents = maxs - mins
    longest = float(extents.max())
    if longest <= 0.0:
        raise ValueError(
            f"STL {stl_path} has zero extent -- all vertices coincide."
        )

    # Uniform scale: longest axis -> body_extent_cells lattice cells.
    scale = body_extent_cells / longest

    # Target placement of the scaled body's bounding-box CORNER in the
    # LBM grid. Inflow buffer = padding_cells[0]; cross-section is
    # centred so the body sits in the middle of (y, z) by default.
    scaled_extents = extents * scale
    pad_x, pad_y, pad_z = padding_cells
    # C-13 (a): refuse to silently clip the body to a sliver. If the
    # scaled bounding box exceeds the available grid on any axis the
    # voxelisation runs anyway and the user gets a 3-cell cross-section
    # of their mesh with no warning -- they'd simulate that cross-section
    # believing it's their whole body. Surface a clear ValueError so the
    # caller can pick a smaller body_extent_cells or larger grid.
    if scaled_extents[0] + pad_x > Nx:
        raise ValueError(
            f"Scaled body extends {scaled_extents[0]:.1f} cells along x "
            f"plus {pad_x} cells inflow padding -- exceeds Nx={Nx}. "
            f"Reduce body_extent_cells or increase Nx."
        )
    if scaled_extents[1] > Ny:
        raise ValueError(
            f"Scaled body extends {scaled_extents[1]:.1f} cells along y "
            f"-- exceeds Ny={Ny}. Reduce body_extent_cells."
        )
    if scaled_extents[2] > Nz:
        raise ValueError(
            f"Scaled body extends {scaled_extents[2]:.1f} cells along z "
            f"-- exceeds Nz={Nz}. Reduce body_extent_cells."
        )
    # Centre of mass of the body, in lattice coords, after placement.
    target_corner = np.array([
        pad_x,
        max(pad_y, 0.5 * (Ny - scaled_extents[1])),
        max(pad_z, 0.5 * (Nz - scaled_extents[2])),
    ], dtype=np.float64)

    # Transform native triangle coords -> lattice coords:
    #   p_lattice = (p_native - mesh_min) * scale + target_corner
    triangles_lattice = (
        (triangles.astype(np.float64) - mins) * scale + target_corner
    )

    mask = voxelize_triangles(
        triangles_lattice.astype(np.float32),
        grid_shape=(Nx, Ny, Nz),
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
    )

    if close_iters > 0:
        mask = binary_closing(mask, iterations=close_iters)

    # Empty / runaway sanity. An empty mask means the mesh fell outside
    # the grid (caller passed a too-small Nx/Ny/Nz or pathological
    # padding). A whole-domain mask means voxelisation flipped parity
    # everywhere -- typically a non-manifold input where the parity
    # rule no longer makes sense.
    n_solid = int(mask.sum())
    if n_solid == 0:
        raise ValueError(
            f"Voxelisation of {stl_path} produced an empty mask. The "
            f"mesh may have landed outside the LBM grid -- try a "
            f"smaller body_extent_cells or smaller padding."
        )
    if n_solid >= mask.size - 1:
        raise ValueError(
            f"Voxelisation of {stl_path} flagged the entire grid as "
            f"solid. The mesh is likely non-manifold; clean it up "
            f"(e.g. meshlab 'Close holes') and try again."
        )

    return mask


def voxel_wall_links(body_mask: np.ndarray):
    """Build a Bouzidi-style wall-link list from a voxel body mask.

    For every fluid cell adjacent to at least one solid cell along a
    D3Q19 lattice direction, emit a ``(x, y, z, dir, q)`` entry where
    ``q`` is the approximate wall fraction in ``(0, 1]`` -- the
    distance from the fluid cell centre to the wall, expressed as a
    fraction of the link length ``|c_dir|``.

    The ``q`` estimate uses linear interpolation on a smoothed copy of
    the mask: smoothing the bool array with a small gaussian
    (sigma = 0.6 cell) produces a scalar field whose 0.5 level set
    approximates the surface to sub-voxel precision. Along each fluid-
    to-solid link we sample the smoothed field at both endpoints and
    solve linearly for the crossing position. The result is sharper
    than halfway BB (which is q = 0.5 by construction) without paying
    the per-link ray-triangle-intersection cost of a triangle-exact
    builder.

    This is the **voxel-side counterpart** of
    :func:`src.lbm_3d_bouzidi.sphere_wall_links`; both return the same
    ``WallLinkList`` dataclass so the LBM kernels can consume either
    transparently. Triangle-exact q is roadmapped as Phase 4 of D-9
    if the smoothed-mask approximation turns out to leave visible Cd
    error in validation.

    Parameters
    ----------
    body_mask : ndarray of shape ``(Nx, Ny, Nz)``, dtype bool
        ``True`` cells are solid. Typically the output of
        :func:`voxel_mask_for_lbm` (or :func:`voxelize_triangles`
        followed by morphological close).

    Returns
    -------
    links : WallLinkList
        Sparse list of fluid-to-solid wall links with sub-voxel q.
    """
    from scipy.ndimage import gaussian_filter

    # Lazy import to avoid a circular dependency between voxelize
    # (sole consumer in the 3D pipeline) and lbm_3d_bouzidi (which
    # owns the WallLinkList dataclass + D3Q19 constants).
    from src.lbm_3d import LATTICE_VELOCITIES_3D
    from src.lbm_3d_bouzidi import WallLinkList

    if body_mask.ndim != 3:
        raise ValueError(
            f"body_mask must be 3-D; got shape {body_mask.shape}."
        )
    if body_mask.dtype != np.bool_:
        body_mask = body_mask.astype(np.bool_, copy=False)

    Nx, Ny, Nz = body_mask.shape
    # Smoothed mask -> proxy for a signed distance field near the
    # surface. sigma = 0.6 is enough to give one cell of crossover at
    # the surface (the 0.5 level set falls inside the boundary voxel
    # row) while still preserving sharp corners.
    smoothed = gaussian_filter(body_mask.astype(np.float32), sigma=0.6)

    # Walk each non-rest lattice direction; for each, identify fluid
    # cells whose neighbour in that direction is in-bounds and solid.
    # Vectorise the per-direction sweep via boolean slicing.
    fluid = ~body_mask

    xs_buf: list[int] = []
    ys_buf: list[int] = []
    zs_buf: list[int] = []
    dirs_buf: list[int] = []
    qs_buf: list[float] = []

    for d_idx in range(1, len(LATTICE_VELOCITIES_3D)):           # skip rest
        cx, cy, cz = (int(v) for v in LATTICE_VELOCITIES_3D[d_idx])
        link_len = float(np.sqrt(cx * cx + cy * cy + cz * cz))

        # Build a fluid-mask view restricted to cells whose neighbour
        # at +c is in-bounds. Using slice ranges keeps the operation
        # branchless and contiguous.
        i_lo = max(0, -cx)
        i_hi = Nx - max(0, cx)
        j_lo = max(0, -cy)
        j_hi = Ny - max(0, cy)
        k_lo = max(0, -cz)
        k_hi = Nz - max(0, cz)
        if i_lo >= i_hi or j_lo >= j_hi or k_lo >= k_hi:
            continue

        fluid_here = fluid[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]
        solid_there = body_mask[
            i_lo + cx:i_hi + cx,
            j_lo + cy:j_hi + cy,
            k_lo + cz:k_hi + cz,
        ]
        wall_link_here = fluid_here & solid_there
        if not wall_link_here.any():
            continue
        ii, jj, kk = np.where(wall_link_here)
        # Translate back to global indices.
        ii_g = ii + i_lo
        jj_g = jj + j_lo
        kk_g = kk + k_lo

        # Smoothed-field samples at both endpoints. The endpoint
        # values bracket 0.5 by construction (fluid endpoint < 0.5,
        # solid endpoint > 0.5 in the bulk; the gaussian preserves
        # this ordering at the surface).
        s_fluid = smoothed[ii_g, jj_g, kk_g]
        s_solid = smoothed[ii_g + cx, jj_g + cy, kk_g + cz]

        # Linear interpolation for the 0.5 crossing parameter
        # t in [0, 1]. The wall fraction q is t scaled by link length
        # (so face-direction links return their natural q, edge
        # directions return q in (0, sqrt(2)] -- the Bouzidi kernel
        # already expects q in fraction-of-link units).
        denom = s_solid - s_fluid
        # Where the smoothed field is locally constant (interior of a
        # block, very flat region), denom is tiny and the linear
        # interpolation is unreliable -- fall back to q = 0.5 (the
        # halfway-BB equivalent) so the kernel stays well-defined.
        safe = np.abs(denom) > 1e-3
        t = np.where(safe, (0.5 - s_fluid) / np.where(safe, denom, 1.0), 0.5)
        # Clip to (0, 1]. A value at the open boundary q = 0 would
        # mean the wall is AT the fluid cell centre, which is a
        # degenerate Bouzidi case (sphere q-generator clips to 1e-3
        # via its smaller-positive-root check). We adopt the same
        # floor here.
        t = np.clip(t, 1e-3, 1.0)
        # q for the WallLinkList is the wall fraction along the
        # link in fraction-of-link units. With t already in [0, 1]
        # and the link spanning length |c|, q = t. The Bouzidi
        # kernel multiplies q by lattice-cell distances internally
        # so no extra scaling is needed here -- consistency with
        # sphere_wall_links (which also returns q in [0, 1]).
        del link_len   # unused; kept for clarity in the comment above

        xs_buf.extend(ii_g.tolist())
        ys_buf.extend(jj_g.tolist())
        zs_buf.extend(kk_g.tolist())
        dirs_buf.extend([d_idx] * len(ii_g))
        qs_buf.extend(t.astype(np.float32).tolist())

    return WallLinkList(
        x=np.asarray(xs_buf, dtype=np.int32),
        y=np.asarray(ys_buf, dtype=np.int32),
        z=np.asarray(zs_buf, dtype=np.int32),
        dir=np.asarray(dirs_buf, dtype=np.int32),
        q=np.asarray(qs_buf, dtype=np.float32),
    )


def voxel_mask_and_links_for_lbm(
    stl_path: str | Path,
    Nx: int,
    Ny: int,
    Nz: int,
    *,
    body_extent_cells: float = 8.0,
    padding_cells: tuple[float, float, float] = (16.0, 12.0, 12.0),
    close_iters: int = 1,
) -> tuple:
    """Aggregate wrapper: STL -> ``(body_mask, wall_links)``.

    Identical placement / scaling semantics as
    :func:`voxel_mask_for_lbm`; just additionally returns the
    Bouzidi-style wall-link list built by :func:`voxel_wall_links`.
    Use this when the consumer wants Bouzidi BB on the uploaded mesh;
    use :func:`voxel_mask_for_lbm` (mask-only) when halfway BB is
    enough.
    """
    mask = voxel_mask_for_lbm(
        stl_path, Nx, Ny, Nz,
        body_extent_cells=body_extent_cells,
        padding_cells=padding_cells,
        close_iters=close_iters,
    )
    links = voxel_wall_links(mask)
    return mask, links


__all__ = [
    "read_stl",
    "voxelize_triangles",
    "voxel_mask_for_lbm",
    "voxel_wall_links",
    "voxel_mask_and_links_for_lbm",
]
