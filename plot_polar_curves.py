import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("sd7003.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()

# Filter for Re = 30000 and AoA between -5 and 15 degrees
mask = (df["Re"] == 30000) & (df["Angle"] >= -5) & (df["Angle"] <= 15)
data = df[mask].sort_values("Angle")

# Average duplicate angles (there are paired measurements at each AoA)
data = data.groupby("Angle", as_index=False).mean(numeric_only=True)

alpha = data["Angle"].values
cl = data["Cl"].values
cd = data["Cd"].values

# Identify stall: Cl peaks then drops
i_clmax = np.argmax(cl)
alpha_stall = alpha[i_clmax]
cl_max = cl[i_clmax]

print(f"Max Cl = {cl_max:.3f} at alpha = {alpha_stall:.1f} deg")
print(f"Stall region: alpha > {alpha_stall:.1f} deg (Cl decreases beyond this point)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Cl vs alpha ---
ax1 = axes[0]
ax1.plot(alpha, cl, "b-o", markersize=5, label=r"$C_l$")
ax1.axvline(alpha_stall, color="r", linestyle="--", alpha=0.7, label=f"Stall onset ({alpha_stall:.1f}°)")
ax1.axvspan(alpha_stall, alpha[-1], color="red", alpha=0.10, label="Stall region")
ax1.scatter([alpha_stall], [cl_max], color="r", zorder=5, s=80,
            label=f"$C_{{l,max}}$ = {cl_max:.3f}")
ax1.set_xlabel(r"Angle of Attack $\alpha$ [°]")
ax1.set_ylabel(r"$C_l$")
ax1.set_title(r"$C_l$ vs $\alpha$ — SD7003, Re = 30 000")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Cd vs alpha ---
ax2 = axes[1]
ax2.plot(alpha, cd, "g-o", markersize=5, label=r"$C_d$")
ax2.axvline(alpha_stall, color="r", linestyle="--", alpha=0.7, label=f"Stall onset ({alpha_stall:.1f}°)")
ax2.axvspan(alpha_stall, alpha[-1], color="red", alpha=0.10, label="Stall region")
ax2.set_xlabel(r"Angle of Attack $\alpha$ [°]")
ax2.set_ylabel(r"$C_d$")
ax2.set_title(r"$C_d$ vs $\alpha$ — SD7003, Re = 30 000")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/sd7003_polar_curves_Re30000.png", dpi=200, bbox_inches="tight")

print("Saved: sd7003_polar_curves_Re30000.png")
