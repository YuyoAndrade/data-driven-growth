from __future__ import division

from functions import *

# import libraries
import pandas as pd
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

show_histogram(tx_user["Recency"], "Recency", "Inactivity Days", "Customers")

# Show the information found, mean, min, max, std, percentiles
print(tx_user.Recency.describe())

# Use Elbow Method to find the optimal amount of clusters
elbow_method(tx_user[["Recency"]])

# Apply KMeans Clustering, goal is to find groups in the data (with an amount given)
# Build 3 clusters for recency and add it to dataframe

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[["Recency"]])
tx_user["RecencyCluster"] = kmeans.predict(tx_user[["Recency"]])

tx_user = order_cluster("RecencyCluster", "Recency", tx_user, False)

tx_user = tx_user.drop(["pat_id"], axis=1)

tx_user = tx_user.groupby(["RecencyCluster"]).agg(
    {"Recency": ["count", "mean", "std", "min", first_qr, median, third_qr, "max"]}
)

tx_user = tx_user.rename(
    columns={"first_qr": "25%", "median": "50%", "third_qr": "75%"}
)

print(tx_user)
