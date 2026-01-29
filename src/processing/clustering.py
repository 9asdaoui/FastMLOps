import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans


def cluster(df: pd.DataFrame):
        
    ### Train K-Means model with chosen K.
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    df['Cluster'] = labels


    ### Add risk_category column (0 = low, 1 = high).
    df['risk_category'] = df['Cluster'].map({
        0: 'High-risk',
        1: 'Low-risk'
    })

    return df