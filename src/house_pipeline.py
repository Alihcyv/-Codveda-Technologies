%%writefile house_modelling.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def preprocessing_house(df: pd.DataFrame) -> tuple:
    print("***Preprocessing house Prediction Data Set.csv***")
    df = df.copy()
    print("Missing value:", df.isnull().sum().sum())
    df.columns = [f"feat_{i}" for i in range(df.shape[1])]
    df.rename(columns={df.columns[-1]: "target_price"}, inplace=True)
    
    X = df.drop('target_price', axis=1)
    y = df['target_price']

    features = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    
    X_train[features] = X_train[features].astype(float)
    X_test[features] = X_test[features].astype(float)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, features

def evaluate_regressor(y_pred, X_test, y_test):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    return r2

house_df = pd.read_csv("/kaggle/input/datasets/elihaciyev/ml-intern/house Prediction Data Set.csv", sep=r'\s+', header=None)
X_train, X_test, y_train, y_test, features = preprocessing_house(house_df)
all_r2_scores = []

print('\n', 30*"#", "Linear Regression", "#"*30, "\n")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = evaluate_regressor(y_pred, X_test, y_test)
all_r2_scores.append(r2)

coeff_df = pd.DataFrame({
    "Features": features,
    "Coefficient": model.coef_
})

coeff_df = coeff_df.sort_values(by="Coefficient", ascending=False)
print("Model Intercept:", model.intercept_)
print("Coefficients Table:")
print(coeff_df)

print("\n", 30*"#", "Decision Tree Regressor", "#"*30, "\n")
model = DecisionTreeRegressor(
    max_depth=3, 
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = evaluate_regressor(y_pred, X_test, y_test)
all_r2_scores.append(r2)


print("\n", 30*"#", "Random Forest Regressor", "#"*30, "\n")
model = RandomForestRegressor(
    n_estimators=500, max_depth=6, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = evaluate_regressor(y_pred, X_test, y_test)
all_r2_scores.append(r2)

importances = model.feature_importances_
features_names = features

importance_df = pd.DataFrame({
    "Features": features_names, "Importance": importances
})
importance_df.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(8, 6))
sns.barplot(
    x='Importance', y='Features', data=importance_df, palette='viridis', hue='Features'
)
plt.title('Feature Importance in Random Forest')
plt.show()

print("\n", 30*"#", "KNeighbors Regressor", "#"*30, "\n")
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = evaluate_regressor(y_pred, X_test, y_test)
all_r2_scores.append(r2)

r2_scores = []
k_range = range(1, 20)

for k in k_range:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2_scores.append(r2_score(y_test, pred))

plt.figure(figsize=(8, 6))
plt.plot(k_range, r2_scores, marker='o', color='g', linestyle='--')
plt.title("Finding optimal k for KNN Regression")
plt.xlabel('Value of k')
plt.ylabel('RMSE (Error)')
plt.xticks(k_range)
plt.grid()
plt.show()

print("\n", 30*"#", "Support Vector Regression", "#"*30, "\n")
svr = SVR(kernel='rbf', C=100.0, epsilon=0.1)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

r2 = evaluate_regressor(y_pred, X_test, y_test)
all_r2_scores.append(r2)

models = ["LR", 'DTR', 'RFR', 'KNNR', "SVR"]

plt.figure(figsize=(8, 6))
plt.bar(models, all_r2_scores, color=['b', 'g', 'y', 'r', 'black'])
plt.ylabel('R2-Score')
plt.title('Comparison of Model R2-Scores')
plt.ylim(0, 1) 
plt.show()
