import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FrequencyAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
df.info()
predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"]==25]["acc_y"].plot()
df[df["set"]==50]["acc_y"].plot()

duration = df[df["set"]==1].index[-1] - df[df["set"]==1].index[0]
duration.total_seconds()

for set_id in df["set"].unique():
    duration = df[df["set"]==set_id].index[-1] - df[df["set"]==set_id].index[0]
    df.loc[df["set"]==set_id, "duration"] = duration.total_seconds()
    
duration_df = df.groupby("category")["duration"].mean()
df.info()


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 5

cutoff = 1.3  # cutoff frequency in Hz

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"]==45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset ["acc_y"].reset_index(drop=True), label= "raw data")
ax [1].plot(subset ["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax [0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax [1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(pca_values) + 1), pca_values, marker="o")
plt.xlabel("Principal Component Number")
plt.ylabel("Explained Variance")
plt.title("PCA Explained Variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, number_comp=3)

subset = df_pca[df_pca["set"]==35]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()
acc_r = np.sqrt(df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2)
gyr_r = np.sqrt(df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2)
df_squared["acc_r"] = acc_r
df_squared["gyr_r"] = gyr_r

subset = df_squared[df_squared["set"]==14]
subset[["acc_r", "gyr_r"]].plot(subplots=True, figsize=(20, 10))

df_squared
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws = 5

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws,"mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws,"std")
    
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(df_temporal, [col], ws,"mean")
        subset = NumAbs.abstract_numerical(df_temporal, [col], ws,"std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list, ignore_index=True)

df_temporal.info()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy()
FreqAbs = FrequencyAbstraction()

fs = 5
ws = int(2800/200)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], fs, ws)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
