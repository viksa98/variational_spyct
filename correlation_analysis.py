import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 1. PREPARE DATA (N, D, VSPYCT performance) FOR EACH TASK

# ---------- Single-Target Regression (STR) -------------
# Datasets: (Name, N, D, VSPYCT_MAE)
str_datasets = [
    ("era", 1000, 5, 1.287840),
    ("cpmp2015", 2108, 27, 918.481147),
    ("qsar234", 2145, 1026, 0.428724),
    ("concrete_compressive_strength", 1030, 9, 7.740371),
    ("yprop", 8885, 252, 0.019969),
    ("puma8NH", 8192, 8, 2.711678),
    ("space_ga", 3107, 6, 0.104501),
    ("bike_sharing", 17379, 6, 11.288880),
    ("quake", 2178, 3, 0.143805),
    ("ailerons", 13750, 40, 0.000127)
]

# ---------- Multi-Target Regression (MTR) -------------
# Datasets: (Name, N, D, VSPYCT_NMAE)
mtr_datasets = [
    ("wq", 1060, 16, 0.183236),
    ("atp1d", 337, 411, 0.125300),
    ("atp7d", 296, 411, 0.167780),
    ("scpf", 1137, 23, 0.027291),
    ("osales", 639, 411, 0.057865),
    ("sf1", 323, 10, 0.092266),
    ("sf2", 1066, 10, 0.037199),
    ("rf1", 9125, 64, 0.166669),
    ("rf2", 9125, 576, 0.193914),
    ("slump", 103, 7, 0.239199)
]

# ---------- Binary Classification (BC) -------------
# Datasets: (Name, N, D, VSPYCT_F1)
bc_datasets = [
    ("scene", 2407, 299, 0.972439),
    ("ova_breast", 1545, 10935, 0.959834),
    ("bioresponse", 3751, 1777, 0.768986),
    ("ova_lung", 1545, 10935, 0.920463),
    ("chronic_kidney_disease", 400, 25, 1.000000),
    ("compas_two_years", 5278, 13, 0.652540),
    ("spambase", 4601, 57, 0.926491),
    ("gina_agnostic", 3468, 970, 0.858579),
    ("credit-g", 1000, 20, 0.681566),
    ("diabetes", 768, 8, 0.715029)
]

# ---------- Multi-Class Classification (MCC) -------------
# Datasets: (Name, N, D, VSPYCT_F1)
mcc_datasets = [
    ("amazon_reviews", 1500, 10000, 0.172791),
    ("dermatology", 366, 34, 0.980117),
    ("micro_mass", 360, 1300, 0.954908),
    ("balance", 625, 4, 0.825152),
    ("yeast", 1484, 8, 0.471005),
    ("wine", 178, 13, 0.966350),
    ("mfeat_zernike", 2000, 47, 0.792283),
    ("har", 10299, 561, 0.984686),
    ("gas_drift", 13910, 128, 0.988948),
    ("semeion", 1593, 257, 0.868518)
]

# 2. Convert these lists into DataFrames
columns_str = ["Dataset", "N", "D", "MAE_VSPYCT"]  # lower is better
str_df = pd.DataFrame(str_datasets, columns=columns_str)

columns_mtr = ["Dataset", "N", "D", "NMAE_VSPYCT"]  # lower is better
mtr_df = pd.DataFrame(mtr_datasets, columns=columns_mtr)

columns_bc = ["Dataset", "N", "D", "F1_VSPYCT"]  # higher is better
bc_df = pd.DataFrame(bc_datasets, columns=columns_bc)

columns_mcc = ["Dataset", "N", "D", "F1_VSPYCT"]  # higher is better
mcc_df = pd.DataFrame(mcc_datasets, columns=columns_mcc)


def corr_and_print(df, perf_col, task_label, higher_is_better=False):
    """
    Compute Spearman correlation between (N, D) and perf_col (the performance measure).
    If higher_is_better, we multiply performance by -1 before correlation
      so that a 'positive correlation' means 'better performance with bigger N or D'.
    """
    x = df[perf_col].values
    if higher_is_better:
        x = -x  # invert to interpret correlation sign consistently
    # correlation with N
    corr_n, p_n = spearmanr(x, df["N"])
    # correlation with D
    corr_d, p_d = spearmanr(x, df["D"])
    print(f"\n[{task_label}] Spearman correlation with N = {corr_n:.3f} (p={p_n:.3f})")
    print(f"[{task_label}] Spearman correlation with D = {corr_d:.3f} (p={p_d:.3f})")


# 3. Calculate & Print correlations for each category
print("=== SINGLE-TARGET REGRESSION (STR) ===")
corr_and_print(str_df, "MAE_VSPYCT", "STR", higher_is_better=False)

print("\n=== MULTI-TARGET REGRESSION (MTR) ===")
corr_and_print(mtr_df, "NMAE_VSPYCT", "MTR", higher_is_better=False)

print("\n=== BINARY CLASSIFICATION (BC) ===")
corr_and_print(bc_df, "F1_VSPYCT", "BC", higher_is_better=True)

print("\n=== MULTI-CLASS CLASSIFICATION (MCC) ===")
corr_and_print(mcc_df, "F1_VSPYCT", "MCC", higher_is_better=True)

