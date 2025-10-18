import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Concrete_Data.csv')
df.columns = df.columns.str.strip()

test_indices = range(500, 630)
df_test = df.iloc[test_indices]
df_train = df.drop(df.index[test_indices])

response_variable = 'Concrete compressive strength(MPa, megapascals)'
predictor_variables = df.columns.drop(response_variable)

X_train = df_train[predictor_variables]
y_train = df_train[response_variable]
X_test = df_test[predictor_variables]
y_test = df_test[response_variable]

#use np to process it with teh buolt in log function
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

X_train_log_const = sm.add_constant(X_train_log)
model_log = sm.OLS(y_train, X_train_log_const).fit()


##evalute on testing data
X_test_log_const = sm.add_constant(X_test_log)
y_pred_test_log = model_log.predict(X_test_log_const)
mse_test_log = mean_squared_error(y_test, y_pred_test_log)
r_squared_test_log = r2_score(y_test, y_pred_test_log)

print("--- Log-Transformed Model Performance on Testing Data ---")
print(f"MSE: {mse_test_log:.2f}")
print(f"Variance Explained (R-squared): {r_squared_test_log:.4f}")

print(model_log.summary())
# print("\n")
# print(model_log.params)