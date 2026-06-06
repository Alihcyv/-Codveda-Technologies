%%writefile churn_pipeline.py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, classification_report


def preprocessing_churn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    print("*** Preprocessing Churn Dataset ***")
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    print(f"Class Imbalance:\n Train:{train_df['Churn'].value_counts()} \nTest:{test_df['Churn'].value_counts()}")
    
    num_columns = [col for col in train_df.columns 
                   if pd.api.types.is_numeric_dtype(train_df[col]) and train_df[col].nunique() > 2]
    
    cat_cols = ['State', 'International plan', 'Voice mail plan']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    
    train_encoded = ohe.fit_transform(train_df[cat_cols])
    test_encoded = ohe.transform(test_df[cat_cols])
     
    cols_names = ohe.get_feature_names_out(cat_cols)
    train_ohe_df = pd.DataFrame(train_encoded, columns=cols_names, index=train_df.index)
    test_ohe_df = pd.DataFrame(test_encoded, columns=cols_names, index=test_df.index)
    
    train_df = pd.concat([train_df, train_ohe_df], axis=1).drop(cat_cols, axis=1)
    test_df = pd.concat([test_df, test_ohe_df], axis=1).drop(cat_cols, axis=1)
    
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df['Churn'])
    test_df['target'] = le.transform(test_df['Churn'])
    
    X_train = train_df.drop(['Churn', 'target'], axis=1)
    y_train = train_df['target']
    X_test = test_df.drop(['Churn', 'target'], axis=1)
    y_test = test_df['target']

    features = X_train.columns
    
    scaler = StandardScaler()
    
    X_train[num_columns] = X_train[num_columns].astype(float)
    X_test[num_columns] = X_test[num_columns].astype(float)
    
    X_train.loc[:, num_columns] = scaler.fit_transform(X_train[num_columns])
    X_test.loc[:, num_columns] = scaler.transform(X_test[num_columns])
    
    return X_train, X_test, y_train, y_test, features

def evaluate_model(y_pred, y_probs, y_test):
    auc_score = roc_auc_score(y_test, y_probs)
    F1_score = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    
    print(f"F1-Score: {F1_score}")
    print(f"Roc-auc-score: {auc_score:.4f}")
    print("\nClassification Report:\n", report)
    return auc_score

def roc_curve_plot(y_probs, y_test, auc_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def pipeline_model(model, name, X_train, X_test, y_train, y_test):
    print(f'\n{30*"#"} {name} {"#"*30}\n')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    auc_score = evaluate_model(y_pred, y_probs, y_test)
    roc_curve_plot(y_probs, y_test, auc_score)

train_df = pd.read_csv("/kaggle/input/datasets/elihaciyev/ml-intern/churn-bigml-80.csv")
test_df = pd.read_csv('/kaggle/input/datasets/elihaciyev/ml-intern/churn-bigml-20.csv')
X_train, X_test, y_train, y_test, features = preprocessing_churn(train_df, test_df)

models_to_test = [
    (LogisticRegression(max_iter=500, C=10.0, class_weight='balanced', random_state=42), "Logistic Regression"),
    (RandomForestClassifier(n_estimators=1000, max_depth=6, class_weight='balanced', random_state=42), "Random Forest Classifier"),
    (SVC(class_weight='balanced', kernel='rbf', probability=True, random_state=42), "Support Vector Machine"),
]

for model_obj, name in models_to_test:
    pipeline_model(model_obj, name, X_train, X_test, y_train, y_test)
