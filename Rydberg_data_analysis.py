# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:08:12 2026

@author: shoha
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("Rydberg Clean 9.csv", header=None)

RUN_COLUMNS = {
    "RUN 1": {"m": 2, "sin": 6},
    "RUN 2": {"m": 10, "sin": 14},
}


rows = []

current_colour = None
current_N = None

for i in df.index:
    row = df.loc[i]
    text = " ".join(row.astype(str)).lower()

    # Metadata
    if "n =" in text:
        current_N = int(text.split("n =")[1].split()[0])
        continue

    if "red" in text:
        current_colour = "red"
        continue
    if "violet" in text:
        current_colour = "violet"
        continue
    if "cyan" in text:
        current_colour = "cyan"
        continue

    # Try extracting data for each run
    for run, cols in RUN_COLUMNS.items():
        try:
            m = int(row[cols["m"]])
            sin_val = float(row[cols["sin"]])
        except:
            continue

        rows.append({
            "m": m,
            "sin_angle_diff": sin_val,
            "run": run,
            "colour": current_colour,
            "N": current_N
        })

df_clean = pd.DataFrame(rows)

sigma_theta = np.radians((224.65-224.4833333)/2)

# 2. Instrumental uncertainty (1 arc minute division)
# We often use half the smallest division for precision, or the full division
# for 'worst-case' instrumental error. Let's use 1 arc min:
sigma_inst = np.radians(1/60) 

# 3. Combined angular uncertainty
sigma_theta_total = np.sqrt(sigma_theta**2 + sigma_inst**2)

d_300 = (1/300)*10**-3
d_80 = (1/80)*10**-3

results = []  # list to store all λ values and info


runs = ["RUN 1", "RUN 2"]          # all runs
colours = ["red", "violet", "cyan"] # all colours
Ns = [80, 300]                     # all N values


for run in runs:
    for colour in colours:
        for N in Ns:
            subset = df_clean[
                (df_clean["run"] == run) &
                (df_clean["N"] == N) &
                (df_clean["colour"] == colour)
            ]

            if subset.empty:
                continue  # skip if no data

            m_vals = subset["m"].values
            sin_vals = subset["sin_angle_diff"].values

            sigma_sin_theta = np.cos(np.arcsin(sin_vals)) * sigma_theta_total
            weights = 1 / sigma_sin_theta

            # Linear fit
            coeffs, cov = np.polyfit(m_vals, sin_vals, deg=1, w=1/sigma_sin_theta, cov=True)
            slope_m_sin = coeffs[0]
            intercept = coeffs[1]
            sigma_slope_m_sin = np.sqrt(cov[0,0])

            lamda = slope_m_sin * (1/(N*1e3))  # if you want λ = slope * d, adjust d as needed
            # If N = 300 lines/mm, d = 1/(N*1e-3)
            
            sigma_lamda = sigma_slope_m_sin * (1/(N*1e3))

            # Save results
            results.append({
                "run": run,
                "colour": colour,
                "N": N,
                "slope": slope_m_sin,
                "intercept": intercept,
                "lambda": lamda,
                "sigma_lambda": sigma_lamda
            })

            # Optional: plot
            plt.errorbar(m_vals, sin_vals, yerr=sigma_sin_theta, fmt='o', capsize=0)
            x_fit = np.linspace(min(m_vals), max(m_vals), 100)
            y_fit = slope_m_sin * x_fit + intercept
            plt.plot(x_fit, y_fit, 'r-', label='Linear fit')
            plt.xlabel("m")
            plt.ylabel("sin(angle diff)")
            plt.title(f"{colour.capitalize()}, {run}, N={N}")
            plt.grid(True)
            plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)
summary_df = results_df.groupby(['colour', 'N']).agg({
    'lambda': 'mean',
    'sigma_lambda': 'mean' # Statistical mean of uncertainties
}).reset_index()

red_n2 = 3
cyan_n2 = 4
violet_n2 = 5

color_to_n2 = {
    "red": red_n2,
    "cyan": cyan_n2,
    "violet": violet_n2
}


summary_df['n2'] = summary_df['colour'].map(color_to_n2)
summary_df["x"] = (1/(summary_df["n2"])**2) - 0.25
summary_df["1/lambda"] = 1 / summary_df["lambda"]

# Level 2 Propagation: sigma_{1/lambda} = sigma_lambda / lambda^2
summary_df["y_err"] = summary_df["sigma_lambda"] / (summary_df["lambda"]**2)

# 3. Final Plots (Example for N=300)
# --- 3. Final Plots (N=300) ---
plt.figure(figsize=(8, 5)) # Create a fresh figure
n300_df = summary_df[summary_df['N'] == 300]
coeffs_300, cov_300 = np.polyfit(n300_df["x"], n300_df["1/lambda"], 1, cov=True)
sigma_R_300 = np.sqrt(cov_300[0,0])

plt.errorbar(n300_df["x"], n300_df["1/lambda"], yerr=n300_df["y_err"], 
             fmt='o', capsize=5, label="Data points", color='orange')
plt.plot(n300_df["x"], coeffs_300[0]*n300_df["x"] + coeffs_300[1], 
         linestyle='--', color='tab:blue', 
         label=f"$R_H = {abs(coeffs_300[0]):.4e} \pm {sigma_R_300:.1e}$")
plt.title("Rydberg Plot for N = 300")
plt.legend()
plt.show()

# --- 4. Final Plots (N=80) ---
plt.figure(figsize=(8, 5)) # Create another fresh figure
n80_df = summary_df[summary_df['N'] == 80]
coeffs_80, cov_80 = np.polyfit(n80_df["x"], n80_df["1/lambda"], 1, cov=True)
sigma_R_80 = np.sqrt(cov_80[0,0])

plt.errorbar(n80_df["x"], n80_df["1/lambda"], yerr=n80_df["y_err"], 
             fmt='o', capsize=5, label="Data points", color='green')
plt.plot(n80_df["x"], coeffs_80[0]*n80_df["x"] + coeffs_80[1], 
         linestyle='--', color='red', 
         label=f"$R_H = {abs(coeffs_80[0]):.4e} \pm {sigma_R_80:.1e}$") # Fixed variables here
plt.title("Rydberg Plot for N = 80")
plt.legend()
plt.show()




R_true = 1.0973731568157e7

# Extract absolute Rydberg values from slopes
RH_300 = abs(coeffs_300[0])
RH_80 = abs(coeffs_80[0])

# Calculate Percentage Differences
perc_diff_300 = (abs(RH_300 - R_true) / R_true) * 100
perc_diff_80 = (abs(RH_80 - R_true) / R_true) * 100

print("="*40)
print(f"{'GRATING RESULTS':^40}")
print("="*40)

print(f"N = 300 lines/mm Grating:")
print(f"  R_H = ({RH_300:.4e} ± {sigma_R_300:.3e}) m^-1")
print(f"  Accuracy: {perc_diff_300:.3f}% difference from true value")

print("-" * 40)

print(f"N = 80 lines/mm Grating:")
print(f"  R_H = ({RH_80:.4e} ± {sigma_R_80:.3e}) m^-1")
print(f"  Accuracy: {perc_diff_80:.3f}% difference from true value")
print("="*40)

print(perc_diff_80/perc_diff_300)
