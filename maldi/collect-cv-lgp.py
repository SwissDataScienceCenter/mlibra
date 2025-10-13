# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Consolidating the first round of CV results
import os
os.listdir()

# %%
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def mean_relative_error(y_true, y_pred):
    """Mean Relative Error per column."""
    # Avoid division by zero
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom), axis=0)

def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE per column."""
    # Avoid division by zero
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom), axis=0) * 100
    
def columnwise_correlation(y_true, y_pred):
    """Pearson correlation per column."""
    corrs = []
    for i in range(y_true.shape[1]):
        corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
        corrs.append(corr)
    return np.array(corrs)
    
def find_target_dirs(root='.'):
    """Find directories matching the pattern and extract FOLD and INDUCING values."""
    pattern = re.compile(r"CV-FOLD-(\d+)-(\d+)-(\d+)-(\d+)_log$")
    targets = []
    for entry in os.listdir(root):
        match = pattern.match(entry)
        if match:
            fold = int(match.group(1))
            latent = int(match.group(2))
            inducing = int(match.group(3))
            batch_size = (int(match.group(4)) - 10) // 100
            targets.append((entry, fold, latent, inducing, batch_size))
    return targets 
    
def compute_metrics(pred_path, true_path):
    y_pred = np.load(pred_path)
    y_true = np.load(true_path)
    assert y_pred.shape == y_true.shape, "Shape mismatch!"
    print(y_pred.shape, y_true.shape)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mre = mean_relative_error(y_true, y_pred)
    corr = columnwise_correlation(y_true, y_pred)
    return mse, mape, mre, corr
    
def collect_metrics(root='/myhome/data/maldi/lmmvae/'):
    records = []
    targets = find_target_dirs(root)
    
    for dirname, fold, latent, inducing, batch_size in targets:
        pred_path = os.path.join(root, dirname, 'test', 'predictions.npy')
        true_path = os.path.join(root, dirname, 'test', 'true_values.npy')
        print(f"loading {pred_path}")
        if not (os.path.exists(pred_path) and os.path.exists(true_path)):
            print(f"skipping{pred_path}")
            continue  # Skip if files are missing
        
        mse, mape, mre, corr = compute_metrics(pred_path, true_path)
        
        # Build record with dynamic column names
        record = {
            'FOLD': fold,
            'INDUCING': inducing,
            'LATENT': latent,
            'BATCH_SIZE': batch_size
        }
        for i, val in enumerate(mse):
            record[f'MSE_{i}'] = val
        for i, val in enumerate(mape):
            record[f'MAPE_{i}'] = val
        for i, val in enumerate(mre):
            record[f'MRE_{i}'] = val
        for i, val in enumerate(corr):
            record[f'CORR_{i}'] = val
            
        records.append(record)
    
    # Build DataFrame
    df = pd.DataFrame(records)
    # Sort columns for readability
    fixed_cols = ['FOLD', 'INDUCING']
    metric_cols = sorted([c for c in df.columns if c not in fixed_cols])
    df = df[fixed_cols + metric_cols]
    return df


# %%
find_target_dirs()

# %%
df_test= collect_metrics()

# %%
df_test


# %%
def plot_metric(df, metric, variable):
    metric_cols = [col for col in df.columns if col.startswith(metric)]
    df_long = df.melt(
        id_vars=["FOLD", variable],
        value_vars=metric_cols,
        var_name= metric+"_Type",
        value_name= metric+"_Value"
    )
    # Cast FOLD as a string for coloring
    df_long["FOLD"] = df_long["FOLD"].astype(str)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x=variable,
        y=metric+"_Value",
        hue="FOLD",
        data=df_long,
        palette="Set2",
        showfliers=False  # This hides the fliers
    )
    plt.title( metric + " by " + variable +", colored by FOLD")
    plt.xlabel(variable)
    plt.ylabel(metric)
    plt.legend(title="FOLD", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    

# %%
plot_metric(df_test, "MSE", "LATENT")

# %%
plot_metric(df_test, "MRE", "LATENT")

# %%
plot_metric(df_test,"CORR", "LATENT")

# %%
plot_metric(df_test,"MSE", "INDUCING")

# %%
plot_metric(df_test,"MRE", "INDUCING")

# %%
plot_metric(df_test,"CORR", "INDUCING")

# %%
plot_metric(df_test,"MSE", "BATCH_SIZE")

# %%
plot_metric(df_test,"MRE", "BATCH_SIZE")

# %%
plot_metric(df_test,"CORR", "BATCH_SIZE")

# %%
corr_columns = [column for column in df_test.columns if "CORR" in column]
mre_columns = [column for column in df_test.columns if "MRE" in column]
mse_columns=[column for column in df_test.columns if "MSE" in column]
average_corr = df_test[corr_columns].mean(axis=1)
average_mre= df_test[mre_columns].mean(axis=1)
average_mse =df_test[mse_columns].mean(axis=1)
df_test['mean_corr']=average_corr.values
df_test['mean_mre'] = average_mre.values
df_test['mean_mse']=average_mse.values
df_test\
.groupby(['BATCH_SIZE', 'INDUCING', 'LATENT']).mean()\
.reset_index()\
.sort_values(by='mean_mse')[['INDUCING','BATCH_SIZE','LATENT','mean_corr','mean_mre','mean_mse']]

# %%
df_test\
.groupby(['BATCH_SIZE', 'INDUCING', 'LATENT']).mean()\
.reset_index()\
.sort_values(by='mean_corr',ascending=False)[['INDUCING','BATCH_SIZE','LATENT','mean_corr','mean_mre','mean_mse']]

# %%
df = df_test
corr_cols = [col for col in df.columns if col.startswith("CORR_")]

# Melt the DataFrame to long format for seaborn
df_long = df.melt(
    id_vars=["FOLD", "LATENT"],
    value_vars=corr_cols,
    var_name="CORR_Type",
    value_name="CORR_Value"
)

# Cast FOLD as a string for coloring
df_long["FOLD"] = df_long["FOLD"].astype(str)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="LATENT",
    y="CORR_Value",
    hue="FOLD",
    data=df_long,
    palette="Set2",
    showfliers=False  # This hides the fliers
)
plt.title("CORR by LATENT, colored by FOLD")
plt.xlabel("LATENT")
plt.ylabel("CORR")
plt.legend(title="FOLD", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
df = df_test
mse_cols = [col for col in df.columns if col.startswith("MSE_")]

# Melt the DataFrame to long format for seaborn
df_long = df.melt(
    id_vars=["FOLD", "LATENT"],
    value_vars=mse_cols,
    var_name="MSE_Type",
    value_name="MSE_Value"
)

# Cast FOLD as a string for coloring
df_long["FOLD"] = df_long["FOLD"].astype(str)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="LATENT",
    y="MSE_Value",
    hue="FOLD",
    data=df_long,
    palette="Set2",
    showfliers=False  # This hides the fliers
)
plt.title("MSE by LATENT, colored by FOLD")
plt.xlabel("LATENT")
plt.ylabel("MSE")
plt.legend(title="FOLD", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
mre_cols = [col for col in df.columns if col.startswith("MRE_")]

# Melt the DataFrame to long format for seaborn
df_long = df.melt(
    id_vars=["FOLD", "INDUCING"],
    value_vars=mre_cols,
    var_name="MRE_Type",
    value_name="MRE_Value"
)

# Cast FOLD as a string for coloring
df_long["FOLD"] = df_long["FOLD"].astype(str)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="INDUCING",
    y="MRE_Value",
    hue="FOLD",
    data=df_long,
    palette="Set2",
    showfliers=False  # This hides the fliers
)
plt.title("MRE by INDUCING, colored by FOLD")
plt.xlabel("INDUCING")
plt.ylabel("MRE")
plt.legend(title="FOLD", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
mape_cols = [col for col in df.columns if col.startswith("MAPE_")]

# Melt the DataFrame to long format for seaborn
df_long = df.melt(
    id_vars=["FOLD", "INDUCING"],
    value_vars=mape_cols,
    var_name="MAPE_Type",
    value_name="MAPE_Value"
)

# Cast FOLD as a string for coloring
df_long["FOLD"] = df_long["FOLD"].astype(str)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="INDUCING",
    y="MAPE_Value",
    hue="FOLD",
    data=df_long,
    palette="Set2",
    showfliers=False  # This hides the fliers
)
plt.title("MAPE by INDUCING, colored by FOLD")
plt.xlabel("INDUCING")
plt.ylabel("MAPE")
plt.legend(title="FOLD", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Load the DataFrame

# Identify all MSE columns
mse_cols = [col for col in df.columns if col.startswith("MSE_")]

# Compute the row-wise mean MSE (across all lipids for each row)
df["mean_MSE"] = df[mse_cols].mean(axis=1)

# Group by INDUCING and take the mean of mean_MSE for each group
inducing_means = df.groupby("INDUCING")["mean_MSE"].mean().reset_index()

# Sort by mean_MSE in ascending order to see which INDUCING is best
inducing_means_sorted = inducing_means.sort_values("mean_MSE", ascending=True)
print(inducing_means_sorted)

# %%
# Identify all MSE columns

# Compute the row-wise mean MSE (across all lipids for each row)
df["mean_MRE"] = df[mre_cols].mean(axis=1)

# Group by INDUCING and take the mean of mean_MSE for each group
inducing_means = df.groupby("INDUCING")["mean_MRE"].mean().reset_index()

# Sort by mean_MSE in ascending order to see which INDUCING is best
inducing_means_sorted = inducing_means.sort_values("mean_MRE", ascending=True)

print(inducing_means_sorted)

# %%
# Compute the row-wise mean MSE (across all lipids for each row)
df["mean_CORR"] = df[corr_cols].mean(axis=1)

# Group by INDUCING and take the mean of mean_MSE for each group
inducing_means = df.groupby("INDUCING")["mean_CORR"].mean().reset_index()

# Sort by mean_MSE in ascending order to see which INDUCING is best
inducing_means_sorted = inducing_means.sort_values("mean_CORR", ascending=False)

print(inducing_means_sorted)

# %%
# Compute the row-wise mean MSE (across all lipids for each row)
df["mean_CORR"] = df[corr_cols].mean(axis=1)

# Group by INDUCING and take the mean of mean_MSE for each group
inducing_means = df.groupby("INDUCING")["mean_"].mean().reset_index()

# Sort by mean_MSE in ascending order to see which INDUCING is best
inducing_means_sorted = inducing_means.sort_values("mean_CORR", ascending=False)

print(inducing_means_sorted)
