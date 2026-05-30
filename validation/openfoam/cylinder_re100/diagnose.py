"""Quick diagnostic of Cd / Cl evolution and Strouhal estimates."""
from pathlib import Path

import numpy as np

fc = Path(__file__).parent / "postProcessing/forceCoeffs/0/forceCoeffs.dat"
data = [
    line.split() for line in fc.read_text().splitlines()
    if line and not line.startswith("#")
]
t = np.array([float(r[0]) for r in data])
cd = np.array([float(r[2]) for r in data])
cl = np.array([float(r[3]) for r in data])

# Per-window means and amplitude (10 evenly-sized windows across full run).
N = 10
chunks = np.array_split(np.arange(len(t)), N)
hdr = ("t_start", "t_end", "Cd_mean", "Cd_amp", "|Cl|_max", "Cl_std")
print(f"{hdr[0]:>9} {hdr[1]:>9} {hdr[2]:>9} {hdr[3]:>9} {hdr[4]:>9} {hdr[5]:>9}")
for ck in chunks:
    if len(ck) < 2:
        continue
    print(
        f"{t[ck[0]]:9.2f} {t[ck[-1]]:9.2f} "
        f"{cd[ck].mean():9.4f} "
        f"{(cd[ck].max()-cd[ck].min()):9.4f} "
        f"{abs(cl[ck]).max():9.4f} "
        f"{cl[ck].std():9.4f}"
    )

D = 2.0
U = 1.0
print()
print("--- Strouhal estimates over last 100 D/U (t = 200 - 400) ---")
mask = t >= 200.0
tt = t[mask]
ll = cl[mask] - cl[mask].mean()
dd = cd[mask] - cd[mask].mean()
dt = float(np.median(np.diff(tt)))

# Method 1: FFT of Cl, pick global max excluding DC + a few bins.
freqs = np.fft.rfftfreq(tt.size, d=dt)
spec_cl = np.abs(np.fft.rfft(ll))
# Skip DC and very-low-freq bins; require f > 0.01 Hz (cuts the envelope drift).
mask_f = freqs > 0.01
peak_idx_cl = np.argmax(spec_cl * mask_f)
f_cl = freqs[peak_idx_cl]
print(f"FFT(Cl) peak: f = {f_cl:.4f} Hz, St = f*D/U = {f_cl*D/U:.4f}")

# Method 2: FFT of Cd (oscillates at 2 * f_shed).
spec_cd = np.abs(np.fft.rfft(dd))
peak_idx_cd = np.argmax(spec_cd * mask_f)
f_cd = freqs[peak_idx_cd]
print(f"FFT(Cd) peak: f = {f_cd:.4f} Hz (= 2 f_shed if Cd doubles), St_implied = {0.5*f_cd*D/U:.4f}")

# Method 3: zero-crossings of Cl (more robust to spectral broadening).
zc = np.where(np.diff(np.signbit(ll)))[0]
if len(zc) > 4:
    # Discard first and last to skip edge effects.
    inner_zc = zc[2:-2]
    inner_t = tt[inner_zc]
    n_full_cycles = (len(inner_t) - 1) / 2.0
    T_zc = (inner_t[-1] - inner_t[0]) / n_full_cycles
    f_zc = 1.0 / T_zc
    print(
        f"Zero-crossing: {len(inner_zc)} crossings, "
        f"{n_full_cycles:.1f} full cycles, T = {T_zc:.2f} s, "
        f"f = {f_zc:.4f} Hz, St = {f_zc*D/U:.4f}"
    )

# Method 4: late window only (t = 300 - 400), where shedding is mostly settled.
print()
print("--- Late window only (t = 300 - 400) ---")
mask_late = t >= 300.0
tt2 = t[mask_late]
ll2 = cl[mask_late] - cl[mask_late].mean()
dt2 = float(np.median(np.diff(tt2)))
freqs2 = np.fft.rfftfreq(tt2.size, d=dt2)
spec_cl2 = np.abs(np.fft.rfft(ll2))
mask_f2 = freqs2 > 0.01
peak2 = np.argmax(spec_cl2 * mask_f2)
print(f"FFT(Cl) peak: f = {freqs2[peak2]:.4f} Hz, St = {freqs2[peak2]*D/U:.4f}")

zc2 = np.where(np.diff(np.signbit(ll2)))[0]
if len(zc2) > 4:
    inner_zc = zc2[1:-1]
    inner_t = tt2[inner_zc]
    n_full_cycles = (len(inner_t) - 1) / 2.0
    T_zc = (inner_t[-1] - inner_t[0]) / n_full_cycles
    print(
        f"Zero-crossing: {len(inner_zc)} crossings, "
        f"{n_full_cycles:.1f} full cycles, T = {T_zc:.2f} s, "
        f"St = {(1/T_zc)*D/U:.4f}"
    )
print(f"\nCd_mean over t=300-400: {cd[t >= 300.0].mean():.4f}")
