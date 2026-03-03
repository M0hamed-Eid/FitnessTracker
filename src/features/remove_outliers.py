import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# Load the processed data
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# Select the first 6 columns (sensor data columns) for outlier detection
outlier_columns = list(df.columns[:6])

# Configure plot styling
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# Create boxplots to visualize data distribution by activity label
df[["gyr_y", "label"]].boxplot(by="label", figsize=(20, 10))

# Boxplots for first 3 sensor columns
df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1,3))
# Boxplots for next 3 sensor columns
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1,3))


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
    dataset (pd.DataFrame): The dataset
    col (string): Column that you want to plot
    outlier_col (string): Outlier column marked with true/false
    reset_index (bool): whether to reset the index for plotting
    """
    # Remove rows with missing values in either column
    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    # Ensure outlier column is boolean type
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    # Reset index if requested for better plotting
    if reset_index:
        dataset = dataset.reset_index()

    # Create plot
    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot normal data points (non-outliers) as blue plus signs
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )

    # Plot outlier data points as red plus signs
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    # Add legend
    plt.legend(
        ["outlier " + col, " no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def mark_outliers_iqr(dataset, col):
    """Mark outliers using the Interquartile Range (IQR) method"""
    dataset = dataset.copy()
    # Calculate quartiles
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR 

    # Mark points outside bounds as outliers
    dataset[col + '_outlier'] = (dataset[col] < lower_bound) | (dataset[col]> upper_bound)

    return dataset

# Test IQR method on a single column
col = "acc_x"
dataset = mark_outliers_iqr(df,col)
plot_binary_outliers(dataset=dataset,col = col, outlier_col = col+"_outlier", reset_index = True)

# Apply IQR method to all columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier", reset_index=True)


# Create histograms to visualize data distribution by label
df[outlier_columns[:3] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout=(3,3))
df[outlier_columns[3:] + ["label"]].plot.hist(by="label", figsize=(20, 10), layout=(3,3))


def mark_outliers_chauvenet(dataset, col, C=2):
    """Mark outliers using Chauvenet's criterion based on normal distribution probability"""
    dataset = dataset.copy()
    
    # Calculate statistics
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)  # Probability threshold

    # Calculate standardized deviations
    deviation = abs(dataset[col] - mean) / std

    # Calculate bounds for probability calculation
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Calculate probability for each data point
    for i in range(0, len(dataset.index)):
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # Mark as outlier if probability below threshold
        mask.append(prob[i] < criterion)

    dataset[col + "_outlier"] = mask
    return dataset

# Apply Chauvenet's method to all columns
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier", reset_index=True)


def mark_outliers_lof(dataset, columns, n=20):
    """Mark outliers using Local Outlier Factor (LOF) algorithm"""
    dataset = dataset.copy()

    # Initialize and apply LOF
    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    # Mark outliers (LOF returns -1 for outliers)
    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

# Apply LOF to all columns simultaneously
dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True)


# Focus on a specific activity: "squat"
label = "squat"

# Apply IQR method to squat data only
for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

# Apply Chauvenet's method to squat data only
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

# Apply LOF to squat data only
dataset, outliers, X_scores = mark_outliers_lof(df[df["label"] == label], outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )


# Test Chauvenet's method on a single column
col = "gyr_z"
dataset = mark_outliers_chauvenet(df, col=col)
# Display outlier rows
dataset[dataset["gyr_z_outlier"]]
# Replace outliers with NaN
dataset.loc[dataset["gyr_z_outlier"], "gyr_z"] = np.nan

# Create a clean dataset with outliers removed
outliers_removed_df = df.copy()
# Remove outliers from all columns, per activity label
for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col)

        # Replace outliers with NaN
        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        # Update the cleaned dataframe
        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = dataset[col]

        # Count and report removed outliers
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label}")

# Save the cleaned dataset
outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")