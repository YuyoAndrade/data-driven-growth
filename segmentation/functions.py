import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Functions to find percentiles (25, 50, 75)

def first_qr(data):
    return data.quantile(0.25)

def median(data):
    return data.quantile(0.5)

def third_qr(data):
    return data.quantile(0.75)

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


def show_histogram(info, title, xlabel, ylabel):
    plot_his = plt.hist(info)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def elbow_method(info):
    sse = {}
    tx_recency = info
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
        tx_recency["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ 

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.show()
