import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Concrete_Data.csv')
df.columns = df.columns.str.strip()

# testing this out to see if ii did in thsi right
print(df.head())

#split data
test_indices = range(500, 630)
df_test = df.iloc[test_indices]
df_train = df.drop(df.index[test_indices])

## Q1.1
response_variable = 'Concrete compressive strength(MPa, megapascals)'
predictor_variables = df.columns.drop(response_variable)

X_train = df_train[predictor_variables]
y_train = df_train[response_variable]
X_test = df_test[predictor_variables]
y_test = df_test[response_variable]

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=predictor_variables, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=predictor_variables, index=X_test.index)

X_train_scaled_const = sm.add_constant(X_train_scaled)

model_scaled = sm.OLS(y_train, X_train_scaled_const).fit()

# # commenting code out for simplicity in the summary printing. These were for other parts of the lab...kinda confusing now...
# r_squared_train = model_scaled.rsquared
# y_pred_train = model_scaled.predict(X_train_const)
# mse_train = mean_squared_error(y_train, y_pred_train)

## testi data
#X_test_const = sm.add_constant(X_test)
# Make predictions on the testing data
#y_pred_test = model_scaled.predict(X_test_const)
# Calculate MSE on the testing data
#mse_test = mean_squared_error(y_test, y_pred_test)
# Calculate R-squared on the testing data
#r_squared_test = r2_score(y_test, y_pred_test)

# print("--- Performance on Training Data ---")
# print(f"MSE: {mse_train:.2f}")
# print(f"Variance Explained (R-squared): {r_squared_train:.4f}")
# print("\n" + "="*40 + "\n")
# print("--- Performance on Testing Data ---")
# print(f"MSE: {mse_test:.2f}")
# print(f"Variance Explained (R-squared): {r_squared_test:.4f}")
# # --- Print the learned parameters ---
# print("Parameters from the 'off-the-shelf' OLS model:")
# print(model.params)


#-------- Q2.1 Priniting summary -----------
print("\n")
print("OLS Model_Scaled Summary Starts NOW:")
print(model_scaled.summary())
print("\n")
print(model_scaled.params)