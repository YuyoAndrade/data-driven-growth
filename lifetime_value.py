from __future__ import division

from segmentation.functions import order_cluster, show_histogram

from datetime import datetime, timedelta, date

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import KFold, cross_val_score, train_test_split

# load our data from CSV
tx_data = pd.read_csv("query_segmentation.csv")

# convert the string date field to datetime
tx_data["created_at"] = pd.to_datetime(tx_data["created_at"]).dt.date
#create 3m and 6m dataframes
tx_3m = tx_data[(tx_data["created_at"] < date(2021,5,1)) & (tx_data["created_at"] >= date(2021,2,1))].reset_index(drop=True)
tx_6m = tx_data[(tx_data["created_at"] >= date(2021,5,1)) & (tx_data["created_at"] < date(2021,11,1))].reset_index(drop=True)

# create a generic user dataframe to keep Patient ID and new segmentation scores
tx_user = pd.DataFrame(tx_3m["pat_id"].unique())
tx_user.columns = ["pat_id"]

# get the max purchase date for each customer and create a dataframe with it
tx_max_purchase = tx_3m.groupby("pat_id").created_at.max().reset_index()
tx_max_purchase.columns = ["pat_id", "MaxPurchaseDate"]

# we take our observation point as the max invoice date in our dataset
tx_max_purchase["Recency"] = (
    tx_max_purchase["MaxPurchaseDate"].max() - tx_max_purchase["MaxPurchaseDate"]
).dt.days

# merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[["pat_id", "Recency"]], on="pat_id")

# Apply KMeans Clustering, goal is to find groups in the data (with an amount given)
kmeans = KMeans(n_clusters=4, max_iter=2000)
kmeans.fit(tx_user[["Recency"]])
tx_user["RecencyCluster"] = kmeans.predict(tx_user[["Recency"]])
tx_user = order_cluster("RecencyCluster", "Recency", tx_user, False)

tx_revenue = tx_data.groupby("pat_id").revenue.sum().reset_index()
tx_revenue.columns = ["pat_id", "Revenue"]

tx_user = pd.merge(tx_user, tx_revenue, on="pat_id")

kmeans = KMeans(n_clusters=4, max_iter=2000)
kmeans.fit(tx_user[["Revenue"]])
tx_user["RevenueCluster"] = kmeans.predict(tx_user[["Revenue"]])
tx_user = order_cluster("RevenueCluster", "Revenue",tx_user,True)

tx_frequency = tx_data.groupby("pat_id").created_at.count().reset_index()
tx_frequency.columns = ["pat_id", "Frequency"]

tx_user = pd.merge(tx_user, tx_frequency, on="pat_id")

kmeans = KMeans(n_clusters=4, max_iter=2000)
kmeans.fit(tx_user[["Frequency"]])
tx_user["FrequencyCluster"] = kmeans.predict(tx_user[["Frequency"]])
tx_user = order_cluster("FrequencyCluster", "Frequency",tx_user,True)

tx_user["OverallScore"] = tx_user["RecencyCluster"] + tx_user["FrequencyCluster"] + tx_user["RevenueCluster"]

tx_score = tx_user

# tx_user = tx_user.groupby("OverallScore")["Recency","Frequency","Revenue"].mean()

tx_score["Segment"] = "Low-Value"
tx_score.loc[tx_score["OverallScore"]>3,"Segment"] = "Mid-Value" 
tx_score.loc[tx_score["OverallScore"]>6,"Segment"] = "High-Value"

print(tx_score)

tx_user_6m = tx_6m.groupby("pat_id")["revenue"].sum().reset_index()
tx_user_6m.columns = ["pat_id","m6_revenue"]

show_histogram(tx_user_6m["m6_revenue"], "Revenue", "Revenue", "Customers")


tx_merge = pd.merge(tx_user, tx_user_6m, on='pat_id', how='left')
tx_merge = tx_merge.fillna(0)

plt.scatter(
    x=tx_merge.query("Segment == 'Low-Value'")["OverallScore"],
    y=tx_merge.query("Segment == 'Low-Value'")["m6_revenue"],
    c="blue",
    edgecolors="black",
    alpha=0.3,
    label="Low-Value"
    ),
plt.scatter(
    x=tx_merge.query("Segment == 'Mid-Value'")["OverallScore"],
    y=tx_merge.query("Segment == 'Mid-Value'")["m6_revenue"],
    c="green",
    edgecolors="black",
    label="Mid-Value",
    ),
plt.scatter(
    x=tx_merge.query("Segment == 'High-Value'")["OverallScore"],
    y=tx_merge.query("Segment == 'High-Value'")["m6_revenue"],
    c="red",
    edgecolors="black",
    label="High-Value",
    ),

plt.xlabel("RFM Score")
plt.ylabel("6m LTV")
plt.title("Lifetime Value")
plt.legend(loc=2)
plt.show()

#remove outliers
tx_merge = tx_merge[tx_merge['m6_revenue']<tx_merge['m6_revenue'].quantile(0.99)]


#creating 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_merge[['m6_revenue']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['m6_revenue']])

#order cluster number based on LTV
tx_merge = order_cluster('LTVCluster', 'm6_revenue',tx_merge,True)

#creatinga new cluster dataframe
tx_cluster = tx_merge.copy()

#see details of the clusters
print(tx_cluster.groupby('LTVCluster')['m6_revenue'].describe())

#convert categorical columns to numerical
tx_class = pd.get_dummies(tx_cluster)

#calculate and show correlations
corr_matrix = tx_class.corr()
print(corr_matrix['LTVCluster'].sort_values(ascending=False))

#create X and y, X will be feature set and y is the label - LTV
X = tx_class.drop(['LTVCluster','m6_revenue'],axis=1)
y = tx_class['LTVCluster']

#split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

gbc_model = GradientBoostingClassifier(max_depth=5, learning_rate=0.1).fit(X_train, y_train)
gbc_model.score(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(gbc_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(gbc_model.score(X_test[X_train.columns], y_test)))

y_pred = gbc_model.predict(X_test)
print(classification_report(y_test, y_pred))
