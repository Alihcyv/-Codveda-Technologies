%%writefile iris_pipeline.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

iris_df = pd.read_csv("/kaggle/input/datasets/elihaciyev/ml-intern/iris.csv")

def preprocessing_iris(df: pd.DataFrame) -> tuple:
    print("*** Preprocessing iris.csv ***")
    df = df.copy()
    
    print("Missing value:", df.isnull().sum().sum())
    print("Imbalance:", df['species'].value_counts())
    
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['species'])
    
    X = df.drop(['target', 'species'], axis=1)
    y = df['target']
    
    features = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, features

f1_scores = []
X_train, X_test, y_train, y_test, features = preprocessing_iris(iris_df)
print("\n", 30*"#", "Logistic Regression", "#"*30, "\n")
model = LogisticRegression(C=1.0, max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = f1_score(y_test, y_pred, average='macro')
f1_scores.append(score)
print(f'F1-Score: {score}')
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n", 30*"#", "KNeighbors Classifier", "#"*30, "\n")
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

score = f1_score(y_test, y_pred, average='macro')
f1_scores.append(score)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f'F1-Score: {score}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

acc = []
k_range = range(1, 15)

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc.append(accuracy_score(y_test, pred))

plt.figure(figsize=(6, 4))
plt.plot(k_range, acc, marker='o', linestyle='--', color='b')
plt.title('Search for Optimal k Value')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy Score')
plt.xticks(k_range)
plt.grid()
plt.show()

print("\n", 30*"#", "Decision Tree Classifier", "#"*30, "\n")
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = f1_score(y_test, y_pred, average='macro')
f1_scores.append(score)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f'F1-Score: {score}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(20,10))
plot_tree(model, 
          feature_names=features, 
          class_names=[str(c) for c in model.classes_], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

print("\n", 30*"#", "Random Forest Classifier", "#"*30, "\n")
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = f1_score(y_test, y_pred, average='macro')
f1_scores.append(score)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f'F1-Score: {score}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

models = ["Logistic Regression", 'KNN', 'Decision Tree', 'Random Forest']

plt.figure(figsize=(6, 4))
plt.bar(models, f1_scores, color=['b', 'g', 'y', 'r'])
plt.ylabel('F1-Score')
plt.title('Comparison of Model F1-Scores')
plt.ylim(0, 1) 
plt.show()
