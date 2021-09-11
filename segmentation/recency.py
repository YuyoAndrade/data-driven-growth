from __future__ import division

from percentiles import *

# import libraries
from datetime import datetime, timedelta
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from chart_studio import plotly
from sklearn.cluster import KMeans

import plotly.graph_objs as go

#load our data from CSV
tx_data = pd.read_csv('query_segmentation.csv')

#convert the string date field to datetime
tx_data['created_at'] = pd.to_datetime(tx_data['created_at'])

#create a generic user dataframe to keep Patient ID and new segmentation scores
tx_user = pd.DataFrame(tx_data['pat_id'].unique())
tx_user.columns = ['pat_id']

#get the max purchase date for each customer and create a dataframe with it
tx_max_purchase = tx_data.groupby('pat_id').created_at.max().reset_index()
tx_max_purchase.columns = ['pat_id','MaxPurchaseDate']

#we take our observation point as the max invoice date in our dataset
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[['pat_id','Recency']], on='pat_id')

plot_his = plt.hist(tx_user["Recency"])

plt.title("Recency")
plt.xlabel("Inactivity Days")
plt.ylabel("Customers")
plt.show()

# Show the information found, mean, min, max, std, percentiles
print(tx_user.Recency.describe())

# Use Elbow Method to find the optimal amount of clusters

sse = {}
tx_recency = tx_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

# Apply KMeans Clustering, goal is to find groups in the data (with an amount given)
# Build 3 clusters for recency and add it to dataframe

kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

# Function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

tx_user = tx_user.drop(["pat_id"], axis=1)

tx_user = tx_user.groupby(["RecencyCluster"]).agg({"Recency": ["count", "mean","std", "min", first_qr, median, third_qr, "max"]})

tx_user = tx_user.rename(columns={"first_qr": "25%", "median": "50%", "third_qr": "75%"})

print(tx_user)
