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

show_histogram(tx_user["Revenue"], "Revenue", "Revenue", "Customers")

print(tx_user.Revenue.describe())

# Use Elbow Method to find the optimal amount of clusters
elbow_method(tx_user[["Revenue"]])

kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_user[["Revenue"]])
tx_user["RevenueCluster"] = kmeans.predict(tx_user[["Revenue"]])

#order the frequency cluster
tx_user = order_cluster("RevenueCluster", "Revenue",tx_user,True)

tx_user.groupby("RevenueCluster")["Revenue"].describe()

tx_user = tx_user.groupby(["RevenueCluster"]).agg({"Revenue": ["count", "mean","std", "min", first_qr, median, third_qr, "max"]})

tx_user = tx_user.rename(columns={"first_qr": "25%", "median": "50%", "third_qr": "75%"})

print(tx_user)