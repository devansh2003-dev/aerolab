"""Airfoil analysis helpers built on AeroSandbox + NeuralFoil.

The single entry point `analyze_airfoil` wraps NeuralFoil so that the rest of the
project never has to know about its exact API. Any future change (caching, swapping
solver, validation) lives here.

Two small input-side helpers (`normalize_naca`, `thickness_camber`) live here
too -- they were extracted from app.py during the D1 split so the Fast-mode
UI in app.py only contains UI code, not airfoil string-parsing logic.
"""
from __future__ import annotations

import aerosandbox as asb
import neuralfoil as nf
import numpy as np

# Type alias: callers can pass either a NACA-name string or a built AeroSandbox airfoil.
AirfoilLike = str | asb.Airfoil


def normalize_naca(raw: str) -> str:
    """Accept '4412', 'naca4412', 'NACA 4412', etc., return 'naca4412'.

    Raises ValueError with a user-actionable message if ``raw`` is not a
    4- or 5-digit NACA code (anything else gets passed straight through
    to asb.Airfoil which silently returns coordinates=None and then
    crashes downstream rendering -- caught in external review 2026-05-24).
    """
    cleaned = raw.strip().lower().replace("naca", "").replace(" ", "")
    if not cleaned.isdigit() or len(cleaned) not in (4, 5):
        raise ValueError(
            f"{raw!r} is not a valid NACA code. Expected 4 or 5 digits "
            f"(e.g. '4412', 'NACA 23012'), got {cleaned!r}."
        )
    return f"naca{cleaned}"


def thickness_camber(coords: np.ndarray, x_stations: np.ndarray):
    """Compute t(x)/c and camber(x)/c at the given x/c stations.

    AeroSandbox airfoil coordinates start at the trailing edge, walk
    counter-clockwise over the upper surface to the leading edge, and
    back to the trailing edge along the lower surface. We split at the
    LE (min-x point) and interpolate each surface at common x stations
    to get thickness and camber.
    """
    le_idx = int(np.argmin(coords[:, 0]))
    upper = coords[: le_idx + 1][::-1]   # LE -> TE on upper surface
    lower = coords[le_idx:]              # LE -> TE on lower surface
    y_upper = np.interp(x_stations, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_stations, lower[:, 0], lower[:, 1])
    thickness = y_upper - y_lower
    camber = (y_upper + y_lower) / 2
    return thickness, camber


def get_airfoil(airfoil: AirfoilLike) -> asb.Airfoil:
    """Build an ``asb.Airfoil`` from a name string, or pass through an existing one.

    Centralizing this conversion means callers don't need to import aerosandbox
    just to construct an airfoil from a NACA name.

    Raises
    ------
    ValueError
        If ``airfoil`` is a string but AeroSandbox can't resolve it to a real
        airfoil (e.g. 'nacabanana', a stray UIUC name with no DAT file). In
        those cases asb.Airfoil silently returns an instance with
        ``coordinates = None`` which then crashes downstream rendering --
        we surface that as an explicit error here so callers can catch it.
    """
    if isinstance(airfoil, str):
        af = asb.Airfoil(airfoil)
        if af.coordinates is None or len(af.coordinates) == 0:
            raise ValueError(
                f"{airfoil!r} is not a recognised airfoil. AeroSandbox "
                f"resolved it to an empty coordinate set. Use a 4-/5-digit "
                f"NACA code (e.g. 'naca4412') or a known UIUC name."
            )
        return af
    return airfoil


def analyze_airfoil(
    airfoil: AirfoilLike,
    alpha,
    Re: float,
    model_size: str = "xxxlarge",
) -> dict:
    """Predict aerodynamic coefficients for an airfoil with NeuralFoil.

    Parameters
    ----------
    airfoil
        Airfoil name (e.g. ``"naca4412"``) or an ``aerosandbox.Airfoil`` instance.
        Strings are passed straight to ``asb.Airfoil(...)``, which understands the
        4- and 5-digit NACA conventions and the UIUC database.
    alpha
        Angle of attack in degrees. Scalar or 1-D array; NeuralFoil is vectorized
        so a single call handles a whole sweep.
    Re
        Reynolds number based on chord.
    model_size
        NeuralFoil model size, trading accuracy for inference time.
        Options (small -> large): ``"xxxsmall"``, ``"xxsmall"``, ``"xsmall"``,
        ``"small"``, ``"medium"``, ``"large"``, ``"xlarge"``, ``"xxlarge"``,
        ``"xxxlarge"``. Default ``"xxxlarge"`` is the most accurate (~1 ms / point).

    Returns
    -------
    dict
        NeuralFoil's aero dict (keys include ``CL``, ``CD``, ``CM``, ``Cpmin``,
        ``Top_Xtr``, ``Bot_Xtr``, mach correction terms) with one extra key
        ``LD`` = ``CL / CD`` added for convenience. All values are numpy arrays.
    """
    airfoil = get_airfoil(airfoil)

    aero = nf.get_aero_from_airfoil(
        airfoil=airfoil,
        alpha=np.asarray(alpha),  # ensure array-like (NeuralFoil is happy either way, but explicit)
        Re=Re,
        model_size=model_size,
    )
    aero["LD"] = aero["CL"] / aero["CD"]
    return aero
