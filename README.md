
# 📊 Customer Segmentation using K-Means Clustering

This project demonstrates **K-Means Clustering** to segment mall customers based on their behavior and demographics using the `Mall_Customers.csv` dataset.

---

## 📁 Dataset
- **Source**: `Mall_Customers.csv`
- **Features**:
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1–100)

---

## 🔍 Workflow Summary

### ✅ Step 1: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
```

---

### ✅ Step 2: Load Dataset
```python
df = pd.read_csv("Mall_Customers.csv")
df.head()
```

---

### ✅ Step 3: Data Preprocessing
```python
# Drop CustomerID
X = df.drop(columns=['CustomerID'])

# Convert categorical column 'Gender'
X = pd.get_dummies(X, drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### ✅ Step 4: Elbow Method for Optimal Clusters
```python
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method For Optimal K")
plt.xlabel("Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("elbow_method.png")
plt.show()
```

---

### ✅ Step 5: Apply KMeans
```python
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_
```

---

### ✅ Step 6: Visualize Clusters (PCA)
```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title("Customer Clusters (PCA 2D)")
plt.savefig("kmeans_clusters.png")
plt.show()
```

---

### ✅ Step 7: Evaluate with Silhouette Score
```python
score = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score: {score:.2f}")
```

---

## 🖼️ Visual Outputs

| Output File           | Description                               |
|------------------------|-------------------------------------------|
| (elbow_method)(elbow_method.png)     | Elbow plot to choose optimal clusters     |
| [kmeans_clusters](kmeans_clusters.png)  | 2D projection of clusters using PCA       |

---

## 🛠️ Technologies Used
- Python 🐍
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (KMeans, PCA, StandardScaler)

---

## 🚀 Run the Notebook
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```
Then open and run: `Task_8_KMeans_Clustering.ipynb`

---

## 💡 Business Impact
Customer segmentation allows businesses to:
- Personalize marketing campaigns
- Target high-value customer groups
- Improve customer retention

---

## 🔗 Connect
Made with ❤️ by [Manish Srivastav](https://www.linkedin.com/in/roxtop07/)
