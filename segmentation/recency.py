# Eugenio Andrade Lozano (euyuyo01@gmail.com)
# Calculate recency, to work use: cd segmentation

from __future__ import division
# import libraries
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from chart_studio import plotly as py

import plotly.offline as pyoff
import plotly.graph_objs as go

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

tx_user.head()

plot_his = plt.hist(tx_user["Recency"])

plt.title("Recency")
plt.xlabel("Inactivity Days")
plt.ylabel("Customers")
plt.show()

