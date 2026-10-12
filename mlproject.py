# Customer Segmentation using K-Means Clustering

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load the dataset
# Replace 'customers.csv' with your dataset file name
data = pd.read_csv("customers.csv")

# 3. Display the first 5 rows
print("First 5 rows:")
print(data.head())

# 4. Check dataset information
print("\nDataset Information:")
print(data.info())

# 5. Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 6. Select features for clustering
# Change these column names according to your dataset
X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# 7. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Find the best number of clusters using Elbow Method
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 9. Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 10. Train K-Means model
# Suppose the elbow suggests 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# 11. Train the model and assign clusters
data["Cluster"] = kmeans.fit_predict(X_scaled)

# 12. Display customers with their cluster
print("\nCustomer Segmentation:")
print(data.head(20))

# 13. Display number of customers in each cluster
print("\nCustomers in Each Cluster:")
print(data["Cluster"].value_counts().sort_index())

# 14. Calculate cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

cluster_centers = pd.DataFrame(
    centers,
    columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
)

print("\nCluster Centers:")
print(cluster_centers)

# 15. Visualize the clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Cluster"],
    s=100
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()

# 16. Save the segmented data
data.to_csv("customer_segments.csv", index=False)

print("\nSegmented data saved as customer_segments.csv")