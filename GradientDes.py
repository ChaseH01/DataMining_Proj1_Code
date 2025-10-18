import numpy as np
import pandas as pd


CSV_PATH = "Concrete_Data.csv"
SCALE = "None"            # options: "standardize", "normalize", or None
ALPHA = 0.000001                   # learning rate (tune this if needed)
EPOCHS = 100000                    # number of GD steps (tune this if needed)

PRED_MATCH = "Age"            
RESP_MATCH = "compressive"       

def find_column(cols, needle):
    """Return the first column whose lowercase name contains `needle`."""
    needle = needle.lower()
    for c in cols:
        if needle in c.lower():
            return c
    raise ValueError(f"Could not find a column containing: {needle!r}. Columns: {list(cols)}")

def standardizer(x):
    mu, sd = np.mean(x), np.std(x, ddof=0)
    sd = sd if sd != 0 else 1.0
    return (lambda v: (v - mu) / sd), (mu, sd)

def normalizer(x):
    mn, mx = np.min(x), np.max(x)
    rng = (mx - mn) if (mx - mn) != 0 else 1.0
    return (lambda v: (v - mn) / rng), (mn, mx)

def mse(y_true, y_pred):
    e = y_true - y_pred
    return float(np.mean(e * e))

def variance_explained(y_true, y_pred):
    # VE ≈ R^2 = 1 - Var(residual)/Var(y)
    num = np.var(y_true - y_pred, ddof=0)
    den = np.var(y_true, ddof=0)
    return float(1 - num / den) if den > 0 else 0.0

def gd_univariate(x, y, alpha=1e-2, epochs=2000):
    w, b = 0.0, 0.0
    n = len(x)
    for _ in range(epochs):
        y_hat = w * x + b
        err = y_hat - y
        # gradients of (1/n) * sum(err^2)
        dw = (2.0 / n) * np.dot(err, x)
        db = (2.0 / n) * np.sum(err)
        w -= alpha * dw
        b -= alpha * db
    return w, b

# ----------------------------
# 3)load data and selcet columns
# ----------------------------
df = pd.read_csv(CSV_PATH)
x_col = find_column(df.columns, PRED_MATCH)
y_col = find_column(df.columns, RESP_MATCH)

# convertint these to numeric arrays
x_all = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
y_all = pd.to_numeric(df[y_col], errors="coerce").to_numpy()

##drop any rows with NaNs (pretty sure these dont exist here but just in case)
mask = ~(np.isnan(x_all) | np.isnan(y_all))
x_all = x_all[mask]
y_all = y_all[mask]

test_idx = np.arange(501, 631)   # inclusive
train_idx = np.setdiff1d(np.arange(len(df)), test_idx)

#### to keep things faithful to the spec we can re-pull by indices from the original df:
x_train_raw = pd.to_numeric(df.loc[train_idx, x_col], errors="coerce").to_numpy()
y_train = pd.to_numeric(df.loc[train_idx, y_col], errors="coerce").to_numpy()
x_test_raw  = pd.to_numeric(df.loc[test_idx,  x_col], errors="coerce").to_numpy()
y_test  = pd.to_numeric(df.loc[test_idx,  y_col], errors="coerce").to_numpy()

#### iff any NaNs slipped in (shouldn't), drop them consistently
train_mask = ~(np.isnan(x_train_raw) | np.isnan(y_train))
test_mask  = ~(np.isnan(x_test_raw)  | np.isnan(y_test))
x_train_raw, y_train = x_train_raw[train_mask], y_train[train_mask]
x_test_raw,  y_test  = x_test_raw[test_mask],  y_test[test_mask]

# ----------------------------
# 4) Scale X only
# --------------------------
if SCALE == "standardize":
    xf, stats = standardizer(x_train_raw)
    x_train = xf(x_train_raw)
    x_test  = xf(x_test_raw)  # TRAIN params
elif SCALE == "normalize":
    xf, stats = normalizer(x_train_raw)
    x_train = xf(x_train_raw)
    x_test  = xf(x_test_raw)  #  TRAIN params
else:
    x_train = x_train_raw.copy().astype(float)
    x_test  = x_test_raw.copy().astype(float)

# ----------------------------
# 5) Train univariate GD and evaluate
# ----------------------------
w, b = gd_univariate(x_train, y_train, alpha=ALPHA, epochs=EPOCHS)

yhat_train = w * x_train + b
yhat_test  = w * x_test + b

mse_train = mse(y_train, yhat_train)
mse_test  = mse(y_test, yhat_test)
ve_train  = variance_explained(y_train, yhat_train)
ve_test   = variance_explained(y_test, yhat_test)

# ----------------------------
# 6) Report
# ----------------------------
print(f"Predictor: {x_col}")
print(f"Response : {y_col}")
print(f"Scaling  : {SCALE or 'none'}\n")

print(f"m (slope) = {w:.6f}")
print(f"b (intercept) = {b:.6f}\n")

print(f"MSE on training data: {mse_train:.6f}")
print(f"Variance Explained / R^2 on training data: {ve_train:.6f}\n")

print(f"MSE on testing data: {mse_test:.6f}")
print(f"Variance Explained / R^2 on testing data: {ve_test:.6f}")
