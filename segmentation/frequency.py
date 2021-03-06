from __future__ import division

from functions import *

# import libraries
import pandas as pd
from sklearn.cluster import KMeans

#load our data from CSV
tx_data = pd.read_csv("query_segmentation.csv")

tx_data['created_at'] = pd.to_datetime(tx_data['created_at'])

tx_user = pd.DataFrame(tx_data['pat_id'].unique())
tx_user.columns = ['pat_id']

tx_frequency = tx_data.groupby('pat_id').created_at.count().reset_index()
tx_frequency.columns = ["pat_id", "Frequency"]

tx_user = pd.merge(tx_user, tx_frequency, on="pat_id")

show_histogram(tx_user["Frequency"], "Frequency", "Frequency", "Customers")

print(tx_user.Frequency.describe())

# Elbow Method 
elbow_method(tx_user[["Frequency"]])

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

#order the frequency cluster
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

#see details of each cluster
tx_user.groupby('FrequencyCluster')['Frequency'].describe()

tx_user = tx_user.groupby(["FrequencyCluster"]).agg({"Frequency": ["count", "mean","std", "min", first_qr, median, third_qr, "max"]})

tx_user = tx_user.rename(columns={"first_qr": "25%", "median": "50%", "third_qr": "75%"})

print(tx_user)
