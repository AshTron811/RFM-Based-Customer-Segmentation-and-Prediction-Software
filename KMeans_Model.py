import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RFMScores = pd.read_csv('RFMScores.csv', encoding='unicode_escape')

scaler = StandardScaler()
KMeansRFM = RFMScores.drop(columns=['RFM_Loyalty_Level'])
Scaled_Data = scaler.fit_transform(KMeansRFM)
KMean_clust = KMeans(n_clusters=10, init='k-means++', max_iter=1000, random_state=42, n_init=10)
KMean_clust.fit(Scaled_Data)
RFMScores['Cluster'] = KMean_clust.labels_

cluster_averages = RFMScores.groupby('Cluster').agg({'R': 'mean', 'F': 'mean', 'M': 'mean'})
def assign_cluster_name(row):
    if row['F'] >= 2.5 and row['R'] >= 2.5 and row['M'] < 2.5:
        return 'Loyal Customers'
    elif row['R'] < 2 and row['F'] >= 3 and row['M'] >= 3:
        return 'At-Risk Customers'
    elif row['R'] >= 1.5 and row['F'] >= 1.5 and row['M'] >= 3 and row['R'] < row['M']:
        return 'Elite Customers'
    elif row['R'] > 3 and row['F'] > 3 and row['M'] > 3:
        return 'High-Value Customers'
    elif row['R'] > 3 and row['F'] < 2:
        return 'New Customers'
    elif row['R'] < 2:
        return 'Churned Customers'
    else:
        return 'Undefined'
cluster_averages['Cluster_Name'] = cluster_averages.apply(assign_cluster_name, axis=1)
RFMScores = RFMScores.merge(cluster_averages, left_on='Cluster', right_index=True, how='left')
RFMScores.rename(columns={'R_x': 'R', 
                         'F_x': 'F', 
                         'M_x': 'M'}, inplace=True)
RFMScores.rename(columns={'R_y': 'R_C.Avg', 
                         'F_y': 'F_C.Avg', 
                         'M_y': 'M_C.Avg'}, inplace=True)

RFMScores['Cluster'] = RFMScores['Cluster_Name'].factorize()[0]

cluster_counts = RFMScores['Cluster'].value_counts()
clusters_to_keep = cluster_counts[cluster_counts > 1000].index.tolist()
RFMScores = RFMScores[RFMScores['Cluster'].isin(clusters_to_keep)]

unique_clusters = sorted(RFMScores['Cluster'].unique())
cluster_mapping = {cluster_label: idx for idx, cluster_label in enumerate(unique_clusters)}
RFMScores['Cluster'] = RFMScores['Cluster'].map(cluster_mapping)

RFMScores.to_csv('RFMScores.csv', index=False)