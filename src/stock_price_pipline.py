%%writefile stock_price_pipline.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def preprocessing_clustering(df):
    
    df.sort_values(['symbol', 'date'], inplace=True)
    df.dropna(inplace=True)
    df['returns'] = df.groupby('symbol')['close'].pct_change()
    df['daily_range'] = (df['high'] - df['low'])/df['close']
    
    stock_profiles = df.groupby('symbol').agg({
        'returns': ['mean', 'std'],           
        'volume': 'mean',                    
        'daily_range': 'mean',               
        'close': lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] 
    })
    
    stock_profiles.columns = [
        'avg_return', 'volatility', 
        'avg_volume', 'avg_daily_range', 
        'total_growth'
    ]
    for col in stock_profiles.columns:
        lower = stock_profiles[col].quantile(0.01)
        upper = stock_profiles[col].quantile(0.99)
        stock_profiles[col] = stock_profiles[col].clip(lower, upper)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(stock_profiles)
    
    df_cluster = pd.DataFrame(X_scaled, index=stock_profiles.index, columns=stock_profiles.columns)
    return df_cluster


df = pd.read_csv("/kaggle/input/datasets/elihaciyev/ml-intern/Stock Prices Data Set.csv")
df_cluster = preprocessing_clustering(df)

print("\n", 30*"#", "KMeans Clustering", "#"*30, "\n")
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_cluster)
df_cluster['cluster'] = clusters

score = silhouette_score(df_cluster, clusters)
print(f"Silhouette Score for {optimal_k} clustering: {score:.4f}\n")

inertia = []
k_range = range(1, 16)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_cluster)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--', color='r')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_range)
plt.grid(True)
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_cluster)

pca_df = pd.DataFrame(
    data=pca_result,
    columns=["PCA1", "PCA2"],
    index=df_cluster.index
)

pca_df['cluster'] = df_cluster['cluster']

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PCA1', 
    y='PCA2', 
    hue='cluster', 
    palette='viridis', 
    data=pca_df, 
    s=50, 
    alpha=0.9,
    edgecolor='w'
)

print(f"\nTotal variance captured by PCA: {sum(pca.explained_variance_ratio_):.2%}\n")
plt.title('Visualization of Stock Clusters using PCA', fontsize=10)
plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=9)
plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=9)
plt.legend(title='Cluster')
plt.grid(True, alpha=0.2)
plt.show()

plt.figure(figsize=(9, 5))
cluster_means_scaled = df_cluster.groupby(df_cluster['cluster']).mean()
sns.heatmap(cluster_means_scaled, annot=True, cmap='RdYlGn')
plt.title('Cluster Characteristics (Standardized)')
plt.xlabel('Features')
plt.xticks(rotation=0) 
plt.ylabel('Cluster ID')
plt.show()
