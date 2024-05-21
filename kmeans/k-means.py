import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

iris = load_iris()
X = iris.data

wcss = []  # Within-cluster sum of squares (WCSS)

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()


# Применение метода силуэта для нахождения оптимального количества кластеров
silhouette_scores = []

for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)


plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Метод силуэта')
plt.xlabel('Количество кластеров')
plt.ylabel('Средний коэффициент силуэта')
plt.show()



kmeans = KMeans(n_clusters=3)
kmeans.fit(X)


fig, ax = plt.subplots(figsize=(8, 6))
colors = ['r', 'g', 'b']
markers = ['o', 's', 'D']

def update(frame):
    ax.clear()
    centroids, clusters = kmeans.history[frame]

    for i, cluster in enumerate(clusters):
        for idx in cluster:
            ax.scatter(X[idx, 0], X[idx, 1], color=colors[i], marker=markers[i])

    for point in centroids:
        ax.scatter(point[0], point[1], s=300, color='k', marker='x')

    ax.set_title(f'Step {frame + 1}')

ani = FuncAnimation(fig, update, frames=len(kmeans.history), interval=500, repeat=False)
plt.show()