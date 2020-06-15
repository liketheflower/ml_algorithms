import numpy as np
import matplotlib.pyplot as plt
N = 4000
K = 3
X = 2
epoch = 20
points = np.random.randn(N, X)

def get_initial_centroids():
    idx = np.arange(N)
    np.random.shuffle(idx)
    return points[idx[:K]]

def get_cluster_id(centroid):
    # points:       5, 2
    # centroids: 3, 1, 2
    # distance:  3, 5, 2
    distance = points - centroid[:, np.newaxis]
    distance = distance**2
    distance = np.sqrt(distance.sum(axis=-1))
    cluster_id = np.argmin(distance, axis=0)
    #https://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes
    total_distance = np.sum(distance[cluster_id, np.arange(N)])
    return cluster_id, total_distance

def update_centroid(cluster_ids):
    centroids = np.zeros((K, X))
    for i in range(K):
        centroids[i] = np.mean(points[cluster_ids==i], axis = 0)
    return centroids

centroids = get_initial_centroids() 
distances = []
for i in range(20):
    cluster_id, total_distance = get_cluster_id(centroids)
    distances.append(total_distance)
    print(total_distance)
    centroids = update_centroid(cluster_id)
plt.plot(distances)
plt.xlabel('iteration')
plt.ylabel('distance')
plt.savefig('iteration_loss'+str(K)+'.png')
plt.show()
plt.scatter(points[:,0], points[:,1], c=cluster_id, s=20,alpha=0.6)
plt.savefig('cluster_'+str(K)+'.png')
plt.show()
