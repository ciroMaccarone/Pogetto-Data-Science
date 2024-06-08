import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

def pca_cluster_3d_visualizator(func, data_clustering, n_clusters, labels, x, y, z, annotations=None, skip_rows=5, annotation_flag=True, title=None, max_iter=300, n_init=10, init='k-means++', random_seed=0):
    km = func(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=random_seed)
    y_means = km.fit_predict(data_clustering)

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n_clusters):
        ax.scatter(data_clustering[y_means == i][:, 0], data_clustering[y_means == i][:, 1], data_clustering[y_means == i][:, 2], s=100, zorder=i, label=f'Group {i} {labels[i]}')

    centers = km.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=50, c='black', alpha=1, marker='*', label='Centroids', zorder=180)

    if annotation_flag and annotations is not None:
        data_clustering = pd.DataFrame(data_clustering)
        ax = show_annotations_3d(annotations, data_clustering, ax, skip_rows=skip_rows)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.mouse_init()
    ax.dist = 6
    plt.title(f'{title}', fontsize=20)
    plt.legend(loc='upper right')
    plt.show()
    return y_means

def show_annotations_3d(annotations, data_clustering, ax, skip_rows=5):
    for index, row in data_clustering.iterrows():
        if index % skip_rows == 0:
            ax.text(row[0], row[1], row[2], annotations.iloc[index][0], fontsize=8, weight='bold', ha='center', zorder=200)
    return ax


def pca_cluster_visualizator(func, data_clustering, n_clusters, labels, x, y, annotations=None, skip_rows=5, annotation_flag=True, title=None, max_iter=300, n_init=10, init='k-means++', random_seed=0):
    km = func(n_clusters=n_clusters, init=init, max_iter=300, n_init=10, random_state=random_seed)
    y_means = km.fit_predict(data_clustering)

    fig = px.scatter()
    
    for i in range(n_clusters):
        cluster_data = data_clustering[y_means == i]
        fig.add_trace(go.Scatter(x=cluster_data[:, 0], y=cluster_data[:, 1], mode='markers',
                                 marker=dict(size=10), name=f'group {i} {labels[i]}'))

    centers = km.cluster_centers_
    fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode='markers',
                             marker=dict(size=10, color='black', symbol='star'), name='centroid'))

    if annotation_flag:
        data_clustering = pd.DataFrame(data_clustering)
        show_annotations(annotations, data_clustering, 0, 1, fig, skip_rows=skip_rows)

    fig.update_layout(title_text=title, xaxis_title=x, yaxis_title=y, legend=dict(x=1, y=1))
    fig.show()
    return y_means

def show_annotations(annotations, data_clustering, x, y, fig, skip_rows=5):
    for index, row in data_clustering.iterrows():
        if index % skip_rows == 0:
            fig.add_trace(go.Scatter(x=[row[x]], y=[row[y]], mode='text', text=[annotations.iloc[index][0]],
                                     textposition='bottom center', textfont=dict(size=10, color='black'), name=f'{annotations.iloc[index][0]}'))

def cluster_visualizator(func, data_clustering, n_clusters, labels, x, y, annotations=None, skip_rows=5, annotation_flag=True, title=None, max_iter=300, n_init=10, init='k-means++', random_seed=0):
    km = func(n_clusters=n_clusters, init=init, max_iter=300, n_init=10, random_state=random_seed)
    y_means = km.fit_predict(data_clustering)

    fig = px.scatter()
    
    for i in range(n_clusters):
        cluster_data = data_clustering[y_means == i]
        fig.add_trace(go.Scatter(x=cluster_data[x], y=cluster_data[y], mode='markers',
                                 marker=dict(size=10), name=f'group {i} {labels[i]}'))

    centroids = km.cluster_centers_
    fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                             marker=dict(size=12, color='black', symbol='star'), name='centroids'))

    if annotation_flag:
        show_annotations(annotations, data_clustering, x, y, fig, skip_rows=skip_rows)

    fig.update_layout(title_text=title, xaxis_title=x, yaxis_title=y, legend=dict(x=1, y=1))
    fig.show()
    return y_means

def hierarchical_cluster_visualizator(func, data_clustering, n_clusters, labels, x, y, annotations=None, skip_rows=5, annotation_flag=True, title=None, affinity='euclidean', linkage='ward'):
    agg_clustering = func(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    y_agg = agg_clustering.fit_predict(data_clustering)

    fig = px.scatter()
    
    for i in range(n_clusters):
        cluster_data = data_clustering[y_agg == i]
        fig.add_trace(go.Scatter(x=cluster_data[x], y=cluster_data[y], mode='markers',
                                 marker=dict(size=10), name=f'group {i} {labels[i]}'))

    if annotation_flag:
        show_annotations(annotations, data_clustering, x, y, fig, skip_rows=skip_rows)

    fig.update_layout(title_text=title, xaxis_title=x, yaxis_title=y, legend=dict(x=1, y=1))
    fig.show()
    return y_agg


def elbow_metod(func, data_clustering, max_iter=300, n_init=10, plot = True, random_seed = 0, init = 'k-means++'):
    wcss = []
    for i in range(1, 11):
        km = func(n_clusters = i, init = init, max_iter = max_iter, n_init = n_init, random_state = random_seed)
        km.fit(data_clustering)
        wcss.append(km.inertia_)
    
    if plot == True:
        elbow_plot(wcss)
    
    return km

def elbow_plot(wcss):
    sns.set(style="darkgrid")
    plt.figure(figsize=(20,8))
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method', fontsize = 20)
    plt.xlabel('No. of Clusters')
    plt.ylabel('wcss')
    plt.show()


def outlier_filter(column, df, threshold=3):
    mean = column.mean()
    std = column.std()
    args = [mean, std, threshold]
    num_of_outliers = column.map(lambda x: 1 if (np.abs(x - args[0]) > args[2] * args[1]) else 0)
    total_outliers = num_of_outliers.sum()
    percentile = total_outliers/len(column)*100
    print(f'filtrati {total_outliers} elementi')
    print(f"Il {percentile}% dei dati si trova al di fuori della soglia di {threshold} deviazioni standard dalla media.")
    df = df[~((column - mean).abs() > threshold * std)]
    return df


def find_optimal_eps(data, min_eps, max_eps, step):
    eps_values = []
    
    # Calcola le distanze tra i punti nel dataset
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    
    # Itera attraverso i valori di eps e calcola il numero di punti nel raggio di eps
    for eps in np.arange(min_eps, max_eps, step):
        num_points = np.sum(distances[:, 1] <= eps)
        distances_within_eps = distances[distances[:, 1] <= eps, 1]
        avg_distance = np.mean(distances_within_eps)
        
        eps_values.append(avg_distance)
    
    # Plotta il grafico delle distanze medie
    fig, _ = plt.subplots(figsize=(20, 8))
    plt.plot(np.arange(min_eps, max_eps, step), eps_values)
    plt.xlabel('Eps')
    plt.ylabel('Average Distance')
    plt.title('Optimal Eps')
    plt.show()
    
    # Trova il valore di eps corrispondente al punto di flessione
    optimal_eps = np.arange(min_eps, max_eps, step)[np.argmax(eps_values)]
    
    return optimal_eps

def plot_silhouette(data, cluster_labels):
    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    plt.rcParams.update({'figure.figsize':(15,8), 'figure.dpi':120})
    fig, ax = plt.subplots()
    y_lower = 10
    for i in np.unique(cluster_labels):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.get_cmap("Spectral")(float(i) / np.max(cluster_labels))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlabel("Valore della silhouette")
    ax.set_ylabel("Etichette del cluster")
    ax.set_yticks([])
    ax.set_title("Grafico della Silhouette")
    plt.show()

def create_3d_figure(x, y, z, x_title="X", y_title="Y", z_title="Z", title="3D Plot"):
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])

    fig.update_layout(scene=dict(
        xaxis_title=x_title,
        yaxis_title=y_title,
        zaxis_title=z_title
    ), title=title)

    fig.show()

def pca_loadings(data_pca, n_components, svd_solver='auto'):
    fig, _ = plt.subplots(figsize=(20, 8))
    pca = PCA(n_components=len(data_pca.columns), svd_solver=svd_solver).fit(data_pca)
    pca_loadings = PCA(n_components=n_components, svd_solver=svd_solver).fit(data_pca)
    loadings = pd.DataFrame(pca_loadings.components_, columns=data_pca.columns)
    explained_variance = pca.explained_variance_ratio_
    print('Explained variance')
    print(explained_variance)
    cumulative_variance = np.cumsum(explained_variance)
    component_labels = data_pca.columns
    plt.bar(component_labels, explained_variance, label='Varianza Spiegata', alpha=0.75)
    plt.plot(component_labels, cumulative_variance, marker='o', color='r', label='Varianza Cumulativa')
    plt.xlabel('Componenti Principali')
    plt.ylabel('Varianza Spiegata')
    plt.legend()
    plt.show()
    print('Carichi (Loadings):')
    print(loadings)

def pca_transform(data_pca, n_components, svd_solver='auto'):
    pca = PCA(n_components=n_components, svd_solver='auto').fit(data_pca)
    data_pca_transformed = pca.transform(data_pca)
    return data_pca_transformed, pca

def plot_average_neighbor_distance(data):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)

    distances = np.sort(distances, axis=0)
    avg_distances = np.mean(distances[:, 1])

    plt.figure(figsize=(12, 8))
    plt.plot(distances[:, 1], label='Distanza dal secondo vicino')
    plt.axhline(y=avg_distances, color='r', linestyle='--', label='Distanza media')
    plt.title('Grafico della Distanza Media dai Vicini più Prossimi')
    plt.xlabel('Punti ordinati per distanza crescente')
    plt.ylabel('Distanza')
    plt.legend()
    plt.show()

def dbscan_cluster_visualizer(data, optimal_eps, x, y, annotations=None, skip_rows=5, title=None, min_samples=5):
    dbscan_model = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    labels = dbscan_model.fit_predict(data)
    data = pd.DataFrame(data)
    # Visualizza il clustering
    fig, ax = plt.subplots(figsize=(20, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # Outliers in nero
            color = 'black'
            label_name = 'Outliers'
        else:
            color = plt.cm.Spectral(label / len(unique_labels))
            label_name = f'Cluster {label}'
            
        ax.scatter(data[labels==label][0], data[labels==label][1], s=50, label=label_name, color=color)

    if annotations is not None:
        data = pd.DataFrame(data)
        ax = plot_annotations(annotations, data, 0, 1, ax, skip_rows=skip_rows)

    sns.set(style="darkgrid")
    plt.rcParams.update({'figure.figsize':(20, 8), 'figure.dpi':120})
    plt.title(f'{title}', fontsize=20)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc='upper right')
    plt.show()

def plot_annotations(annotations, data_clustering, x, y, plt_obj, skip_rows=5):
    for index, row in data_clustering.iterrows():
        if index % skip_rows == 0:
            plt_obj.annotate(annotations.iloc[index][0], (row[x], row[y]), textcoords="offset points", xytext=(5,5), ha='center', weight='bold', fontsize=10, rotation=45)
    return plt_obj


def dbscan_3d_visualizer(data, optimal_eps, x, y, z, annotations=None, skip_rows=5, title=None, min_samples=5):
    dbscan_model = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    labels = dbscan_model.fit_predict(data)
    data = pd.DataFrame(data)
    
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # Outliers in nero
            color = 'black'
            label_name = 'Outliers'
        else:
            color = plt.cm.Spectral(label / len(unique_labels))
            label_name = f'Cluster {label}'
            
        ax.scatter(data[labels==label][0], data[labels==label][1], data[labels==label][2], s=100, label=label_name, color=color)

    if annotations is not None:
        ax = show_annotations_3d(annotations, data, ax, skip_rows=skip_rows)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.mouse_init()
    ax.dist = 6
    plt.title(f'{title}', fontsize=20)
    plt.legend(loc='upper right')
    plt.show()


'''
def cluster_visualizator(func, data_clustering, n_clusters, labels, x, y, annotations=None, skip_rows=5, annotation_flag=True, title=None, max_iter=300, n_init=10, init='k-means++', random_seed=0):
    km = func(n_clusters=n_clusters, init=init, max_iter=300, n_init=10, random_state=random_seed)
    y_means = km.fit_predict(data_clustering)
    with plt.ion():
        fig, ax = plt.subplots(figsize=(20, 8))
        for i in range(n_clusters):
            ax.scatter(data_clustering[y_means == i][x], data_clustering[y_means == i][y], s=100, label=f'group {i} {labels[i]}')
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='black', marker='*', label='centroid')
        if annotation_flag:
            ax = show_annotations(annotations, data_clustering, x, y, ax, skip_rows=skip_rows)
        sns.set(style="darkgrid")
        plt.title(f'{title}', fontsize=20)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend(loc='upper right')
        #fig = mpl_to_interactive(fig, ax)
        fig.show()

 def hierarchical_cluster_visualizator(func,  data_clustering, n_clusters, labels, x, y, annotations=None, skip_rows=5, annotation_flag=True, title=None, affinity='euclidean', linkage='ward'):
    agg_clustering = func(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    y_agg = agg_clustering.fit_predict(data_clustering)
    fig, ax = plt.subplots(figsize=(20, 8))
    for i in range(n_clusters):
        ax.scatter(data_clustering[y_agg == i][x], data_clustering[y_agg == i][y], s=100, label=f'group {i} {labels[i]}')
    if annotation_flag:
        ax = show_annotations(annotations, data_clustering, x, y, ax, skip_rows=skip_rows)
    sns.set(style="darkgrid")
    plt.rcParams.update({'figure.figsize':(20,8), 'figure.dpi':120})
    plt.title(f'{title}', fontsize=20)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc='upper right')
    plt.show()

def pca_cluster_visualizator(func, data_clustering, n_clusters, labels, x, y, annotations=None, skip_rows=5, annotation_flag=True, title=None, max_iter=300, n_init=10, init='k-means++', random_seed=0):
    km = func(n_clusters = n_clusters, init = init, max_iter = 300, n_init = 10, random_state = random_seed)
    y_means = km.fit_predict(data_clustering)
    fig, ax = plt.subplots(figsize=(20, 8))
    for i in range(n_clusters):
        ax.scatter(data_clustering[y_means == i][:, 0], data_clustering[y_means == i][:, 1], s=50, label=f'group {i} {labels[i]}')
    centers = km.cluster_centers_
    ax.scatter(centers[:,0], centers[:, 1], s = 50, c = 'black' , marker='*', label = 'centroid')
    if annotation_flag:
        data_clustering = pd.DataFrame(data_clustering)
        ax = show_annotations(annotations, data_clustering, 0, 1, ax, skip_rows=skip_rows)
    sns.set(style="darkgrid")
    plt.rcParams.update({'figure.figsize':(20,8), 'figure.dpi':120})
    plt.title(f'{title}', fontsize = 20)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc='upper right')
    plt.show()
'''