%%writefile iris_modelling.py
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
iris_df = pd.read_csv("/kaggle/input/datasets/elihaciyev/ml-intern/iris.csv")
X_train, X_test, y_train, y_test, features = preprocessing_iris(iris_df)
models_to_test = [
    (LogisticRegression(C=1.0, max_iter=1000), "Logistic Regression"),
    (KNeighborsClassifier(n_neighbors=7), "KNeighbors Classifier"),
    (DecisionTreeClassifier(max_depth=3, random_state=42), "Decision Tree Classifier"),
    (RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42), "Random Forest Classifier")
]

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    print("\n", 30*"#", model_name, "#"*30, "\n")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    score = f1_score(y_test, y_pred, average='macro')
    print(f'F1-Score: {score}')
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return score

def optimize_knn(X_train, X_test, y_train, y_test):
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

f1_scores = []
for model_obj, name in models_to_test:
    score = evaluate_model(model_obj, name, X_train, X_test, y_train, y_test)
    f1_scores.append(score)
    if name == 'KNeighbors Classifier':
        optimize_knn(X_train, X_test, y_train, y_test)
        

model_names = ['Log_Reg', 'KNN', 'Dec_tree_class', 'Random_forest']
plt.figure(figsize=(6, 4))
plt.bar(model_names, f1_scores, color=['b', 'g', 'y', 'r'])
plt.ylabel('F1-Score')
plt.title('Comparison of Model F1-Scores')
plt.ylim(0, 1) 
plt.show()
