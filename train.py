import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

df=pd.read_csv("Mall_Customers.csv")
x=df[["Annual Income (k$)","Spending Score (1-100)"]].values
k=KMeans(n_clusters=5,random_state=10)
k.fit(x)
L=k.labels_
print(L)
c=k.cluster_centers_
print(c)
plt.scatter(x[:,0],x[:,1],c=L)
plt.scatter(c[:,0],c[:,1],color="red",s=100)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall customer analysis")
plt.close()

# Save the model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(k, f)
print("Model saved as kmeans_model.pkl")