import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = os.path.join(base_dir, "data/interim/02_outliers_removed_chauvenets.pkl")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle(file_path)
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

df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot(subplots=True, figsize=(20, 10))
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot(subplots=True, figsize=(20, 10))
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index
FreqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], fs, ws)

# visualize results
subset = df_freq[df_freq["set"]==15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot(subplots=True, figsize=(20, 10))

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}")
    subset = df_freq[df_freq["set"]==s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columnns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columnns]
    kmeans = KMeans(n_clusters=k,n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)
subset = df_cluster[cluster_columnns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# plot clusters
plt.figure(figsize=(10, 10))
for i, col in enumerate(cluster_columnns):
    plt.subplot(1, 3, i+1)
    for cluster in df_cluster["cluster"].unique():
        subset = df_cluster[df_cluster["cluster"]==cluster]
        plt.scatter(subset[col], subset["cluster"], label=f"Cluster {cluster}")
    plt.xlabel(col)
    plt.ylabel("Cluster")
    plt.title(f"{col} vs Cluster")
    plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle(os.path.join("../../data/interim/03_data_features.pkl"))