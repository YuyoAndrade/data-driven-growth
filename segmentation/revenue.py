from __future__ import division

from functions import *

# import libraries
from datetime import datetime, timedelta
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from chart_studio import plotly
from sklearn.cluster import KMeans

#load our data from CSV
tx_data = pd.read_csv("query_segmentation.csv")

tx_data['created_at'] = pd.to_datetime(tx_data['created_at'])

tx_user = pd.DataFrame(tx_data['pat_id'].unique())
tx_user.columns = ['pat_id']

tx_revenue = tx_data.groupby('pat_id').revenue.sum().reset_index()
tx_revenue.columns = ["pat_id", "Revenue"]

tx_user = pd.merge(tx_user, tx_revenue, on="pat_id")

plot_his = plt.hist(tx_user["Revenue"])

plt.title("Revenue")
plt.xlabel("Revenue")
plt.ylabel("Customers")
plt.show()

print(tx_user.Revenue.describe())

# Elbow Method
sse = {}
tx_recency = tx_user[["Revenue"]]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_user[["Revenue"]])
tx_user["RevenueCluster"] = kmeans.predict(tx_user[["Revenue"]])

#order the frequency cluster
tx_user = order_cluster("RevenueCluster", "Revenue",tx_user,True)

tx_user.groupby("RevenueCluster")["Revenue"].describe()

tx_user = tx_user.groupby(["RevenueCluster"]).agg({"Revenue": ["count", "mean","std", "min", first_qr, median, third_qr, "max"]})

tx_user = tx_user.rename(columns={"first_qr": "25%", "median": "50%", "third_qr": "75%"})

print(tx_user)