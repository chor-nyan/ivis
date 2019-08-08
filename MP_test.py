# import hub_toolbox
# from sklearn.datasets import fetch_openml
# from hub_toolbox.distances import euclidean_distance
#
# mnist = fetch_openml('mnist_784', version=1)
# vectors = mnist.data
#
# vectors = vectors[:1000, :]
# d_mle = hub_toolbox.intrinsic_dimension.intrinsic_dimension(vectors)
# # vectors = vectors[:10000, :]
# # d_mle = hub_toolbox.intrinsic_dimension.intrinsic_dimension(vectors)
# # vectors = mnist.data
# # vectors = vectors[:10000, :]
# # d_mle = hub_toolbox.intrinsic_dimension.intrinsic_dimension(vectors)
# D = euclidean_distance(vectors)
#
# S_k, _, _ = hub_toolbox.hubness.hubness(D=D, k=5, metric='distance')
# D_mp = hub_toolbox.global_scaling.mutual_proximity_empiric(
#         D=D, metric='distance')
# S_k_mp, _, _ = hub_toolbox.hubness.hubness(D=D_mp, k=5, metric='distance')
#
# print(S_k, S_k_mp)


from skhubness.data import load_dexter

X, y = load_dexter()

from skhubness import Hubness
hub = Hubness(k=10, metric='cosine')
hub.fit(X)
k_skew = hub.score()
print(f'Skewness = {k_skew:.3f}')

from skhubness.neighbors import kneighbors_graph
k = 5
# neigbor_graph = kneighbors_graph(X, n_neighbors=k, hubness='mutual_proximity')
neigbor_graph = kneighbors_graph(X, n_neighbors=k, hubness=None)
neighbor_matrix = neigbor_graph.indices.reshape((X.shape[0], k))
print(neighbor_matrix)