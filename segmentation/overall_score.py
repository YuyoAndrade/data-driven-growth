from __future__ import division

from functions import *

# import libraries
from datetime import datetime, timedelta
import pandas as pd

# %matplotlib inline
import numpy as np
import seaborn as sns
from chart_studio import plotly
from sklearn.cluster import KMeans

# load our data from CSV
tx_data = pd.read_csv("query_segmentation.csv")

# convert the string date field to datetime
tx_data["created_at"] = pd.to_datetime(tx_data["created_at"])

# create a generic user dataframe to keep Patient ID and new segmentation scores
tx_user = pd.DataFrame(tx_data["pat_id"].unique())
tx_user.columns = ["pat_id"]

# get the max purchase date for each customer and create a dataframe with it
tx_max_purchase = tx_data.groupby("pat_id").created_at.max().reset_index()
tx_max_purchase.columns = ["pat_id", "MaxPurchaseDate"]

# we take our observation point as the max invoice date in our dataset
tx_max_purchase["Recency"] = (
    tx_max_purchase["MaxPurchaseDate"].max() - tx_max_purchase["MaxPurchaseDate"]
).dt.days

# merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[["pat_id", "Recency"]], on="pat_id")

# Apply KMeans Clustering, goal is to find groups in the data (with an amount given)
# Build 3 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=3)

kmeans.fit(tx_user[["Recency"]])
tx_user["RecencyCluster"] = kmeans.predict(tx_user[["Recency"]])

tx_revenue = tx_data.groupby("pat_id").revenue.sum().reset_index()
tx_revenue.columns = ["pat_id", "Revenue"]

tx_user = pd.merge(tx_user, tx_revenue, on="pat_id")

kmeans.fit(tx_user[["Revenue"]])
tx_user["RevenueCluster"] = kmeans.predict(tx_user[["Revenue"]])

tx_frequency = tx_data.groupby("pat_id").created_at.count().reset_index()
tx_frequency.columns = ["pat_id", "Frequency"]

tx_user = pd.merge(tx_user, tx_frequency, on="pat_id")

kmeans.fit(tx_user[["Frequency"]])
tx_user["FrequencyCluster"] = kmeans.predict(tx_user[["Frequency"]])

tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user = tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()

print(tx_user)