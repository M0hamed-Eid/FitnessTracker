import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])
plt.show()
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

df["label"].unique()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Plot Compare Accelerometer Signals Across Exercises in one figure
# --------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

# Loop through each exercise label
for label in df["label"].unique():

    subset = df[df["label"] == label]

    # Take the first 100 samples for comparison
    signal = subset["acc_y"].iloc[:100].reset_index(drop=True)

    # Plot on the same axis
    ax.plot(signal, label=label)

# Labels and formatting
ax.set_title("Accelerometer Y-axis Comparison Across Exercises")
ax.set_xlabel("Samples")
ax.set_ylabel("Acceleration (acc_y)")
ax.legend()
ax.grid(True)

plt.show()
output_dir = "../../Reports/Figures"
fig.savefig(
    f"{output_dir}/Compare Accelerometer Signals Across Exercises.png",
    dpi=300,
    bbox_inches="tight",
)
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.rcParams["figure.figsize"] = (20, 5)

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()
plt.grid(visible=True)

import os

output_dir = "../../Reports/Figures"
os.makedirs(output_dir, exist_ok=True)

fig.savefig(
    f"{output_dir}/squat_A_category_comparison.png", dpi=300, bbox_inches="tight"
)
plt.close(fig)

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()
plt.grid(visible=True)

fig.savefig(f"{output_dir}/participant_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)


fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()

fig.savefig(f"{output_dir}/Plot multiple axis.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
