import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

outdir = Path("figures")
outdir.mkdir(parents=True, exist_ok=True)

tau = np.logspace(0, 3, 100)
ds_smallworld = 3.0 + 0.5 * (1 - np.exp(-tau/50))
ds_void = 3.0 - 0.15 * (1 - np.exp(-tau/20)) - 0.1 * np.sin(np.log(tau)) * 0.2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.semilogx(tau, ds_smallworld, linewidth=2, label="High Connectivity (Halo)")
ax1.semilogx(tau, ds_void, linestyle="--", linewidth=2, label="Percolation Cluster (Void)")
ax1.axhline(3.0, linestyle=":", alpha=0.7)
ax1.set_xlabel(r"Graph Diffusion Time $\tau$")
ax1.set_ylabel(r"Spectral Dimension $d_S(\tau)$")
ax1.set_title("(A) Effective Dimension Flow")
ax1.legend()
ax1.grid(True, which="both", alpha=0.2)

ds_vals = np.linspace(2.0, 4.0, 100)
c0 = 1.0
gamma = 1.5
c_eff = c0 * (ds_vals / 3.0)**gamma

ax2.plot(ds_vals, c_eff, linewidth=2)
ax2.scatter([3.0], [1.0], zorder=5, label=r"Standard Vacuum ($d_S=3$)")
ax2.scatter([2.85], [c0*(2.85/3.0)**gamma], zorder=5, label=r"Void Regime ($d_S<3$)")
ax2.set_xlabel(r"Spectral Dimension $d_S$")
ax2.set_ylabel(r"Effective Speed $c_{\mathrm{eff}} / c_0$")
ax2.set_title("(B) Propagation Ansatz")
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(outdir / "toy_drnd_ds_ceff.pdf")
plt.savefig(outdir / "toy_drnd_ds_ceff.png", dpi=200)
plt.close(fig)
print("Wrote figures/toy_drnd_ds_ceff.pdf and .png")
