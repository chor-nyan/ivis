"""
Triplet generators.

Functions for creating generators that will yield batches of triplets.

Triplets will be created using KNNs, which can either be precomputed or
dynamically generated.
- generate_knn_triplets_from_neighbour_list will precompute KNNs
- generate_knn_triplets_from_annoy_index will dynamically generate KNNs

Where possible, precomputed KNNs are advised for speed, but where memory is
a concern, dynamically generated triplets can be useful.

"""

import numpy as np
from .knn import extract_knn
from annoy import AnnoyIndex
from keras.utils import Sequence
from scipy.sparse import issparse
from numba import jit
from skhubness.neighbors import kneighbors_graph
# from scipy.spatial.distance import cdist, pdist, squareform
# import hub_toolbox
# from sklearn.datasets import fetch_openml
# from hub_toolbox.distances import euclidean_distance


# def euclidean_distance(X):
#     """Calculate the euclidean distances between all pairs of vectors in `X`.
#
#     Consider using sklearn.metric.pairwise.euclidean_distances for faster,
#     but less accurate distances (not necessarily symmetric, too)."""
#     return squareform(pdist(X, 'euclidean'))

@jit
def make_mutual(neighbour_matrix):
    for i in range(neighbour_matrix.shape[0]):
        for j in range(neighbour_matrix.shape[1]):
            if i not in neighbour_matrix[neighbour_matrix[i, j]]:
                neighbour_matrix[i, j] = neighbour_matrix.shape[0] + 1
    return neighbour_matrix

def generator_from_index(X, Y, index_path, k, batch_size, search_k=-1,
                         precompute=True, verbose=1, type='tri', knn='MP'):
    if k >= X.shape[0] - 1:
        raise Exception('''k value greater than or equal to (num_rows - 1)
                        (k={}, rows={}). Lower k to a smaller
                        value.'''.format(k, X.shape[0]))
    if batch_size > X.shape[0]:
        raise Exception('''batch_size value larger than num_rows in dataset
                        (batch_size={}, rows={}). Lower batch_size to a
                        smaller value.'''.format(batch_size, X.shape[0]))

    if Y is None:
        if precompute:
            if verbose > 0:
                print('Extracting KNN from index')

            if knn == 'MP':
                if verbose > 0:
                    print('Making MP-based KNN')

                # D = euclidean_distance(X)
                # D_mp = hub_toolbox.global_scaling.mutual_proximity_empiric(
                #     D=D, metric='distance')
                neigbour_graph = kneighbors_graph(X, n_neighbors=k, hubness='mutual_proximity', hubness_params={'method': 'normal'})
                # neigbor_graph = kneighbors_graph(X, n_neighbors=k, hubness=None)
                neighbour_matrix = neigbour_graph.indices.reshape((X.shape[0], k))

            else:
                neighbour_matrix = extract_knn(X, index_path, k=k,
                                           search_k=search_k, verbose=verbose)
            # neighbour_matrix = np.asarray(neighbour_matrix, dtype=np.int32)
            print('neighbour_matrix: ', neighbour_matrix.shape)

            if knn == 'Mutual':
                if verbose > 0:
                    print('Making KNN mutual')

                neighbour_matrix = make_mutual(neighbour_matrix)



            # print('Mutual Knn: ', neighbour_matrix[0])

            if type == 'quad':
                return KnnQuadrupletGenerator(X, neighbour_matrix, batch_size=batch_size)
            if type == 'tri':
                return KnnTripletGenerator(X, neighbour_matrix,
                                       batch_size=batch_size)
        else:
            index = AnnoyIndex(X.shape[1])
            index.load(index_path)
            return AnnoyTripletGenerator(X, index, k=k,
                                         batch_size=batch_size,
                                         search_k=search_k)
    else:
        if precompute:
            if verbose > 0:
                print('Extracting KNN from index')

            neighbour_matrix = extract_knn(X, index_path, k=k,
                                           search_k=search_k, verbose=verbose)
            return LabeledKnnTripletGenerator(X, Y, neighbour_matrix,
                                              batch_size=batch_size)
        else:
            index = AnnoyIndex(X.shape[1])
            index.load(index_path)
            return LabeledAnnoyTripletGenerator(X, Y, index,
                                                k=k, batch_size=batch_size,
                                                search_k=search_k)


class AnnoyTripletGenerator(Sequence):

    def __init__(self, X, annoy_index, k=150, batch_size=32, search_k=-1):
        self.X = X
        self.annoy_index = annoy_index
        self.k = k
        self.batch_size = batch_size
        self.search_k = search_k
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size,
                              min((idx + 1) * self.batch_size, self.X.shape[0]))

        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        triplet_batch = [self.knn_triplet_from_annoy_index(row_index)
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return ([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]],
                placeholder_labels)

    def knn_triplet_from_annoy_index(self, row_index):
        """ A random (unweighted) positive example chosen. """
        triplet = []
        neighbour_list = np.array(self.annoy_index.get_nns_by_item(row_index, self.k + 1,
                                  search_k=self.search_k, include_distances=False), dtype=np.uint32)

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        negative_ind = np.random.randint(0, self.X.shape[0])  # Pick a random index until one fits constraint. An optimization.
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplet += [self.X[row_index], self.X[neighbour_ind], self.X[negative_ind]]
        return triplet


class KnnTripletGenerator(Sequence):

    def __init__(self, X, neighbour_matrix, batch_size=32):
        self.X = X
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0]))

        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        triplet_batch = [self.knn_triplet_from_neighbour_list(row_index, self.neighbour_matrix[row_index])
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]
        triplet_batch = np.array(triplet_batch)

        return ([triplet_batch[:, 0],
                 triplet_batch[:, 1],
                 triplet_batch[:, 2]],
                placeholder_labels)

    def knn_triplet_from_neighbour_list(self, row_index, neighbour_list):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplets += [self.X[row_index], self.X[neighbour_ind],
                     self.X[negative_ind]]
        return triplets

class KnnQuadrupletGenerator(Sequence):

    def __init__(self, X, neighbour_matrix, batch_size=32):
        self.X = X
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0]))

        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        quadruplet_batch = [self.knn_quadruplet_from_neighbour_list(row_index, self.neighbour_matrix)
                         for row_index in batch_indices]

        if (issparse(self.X)):
            quadruplet_batch = [[e.toarray()[0] for e in t] for t in quadruplet_batch]
        quadruplet_batch = np.array(quadruplet_batch)

        return ([quadruplet_batch[:, 0],
                 quadruplet_batch[:, 1],
                 quadruplet_batch[:, 2],
                 quadruplet_batch[:, 3]],
                placeholder_labels)

    def knn_quadruplet_from_neighbour_list(self, row_index, neighbour_matrix):
        """ A random (unweighted) positive example chosen. """
        quadruplets = []

        neighbour_list = neighbour_matrix[row_index]

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(list(set(neighbour_list) & set(range(self.X.shape[0]))))

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        neighbour_list_neg = neighbour_matrix[negative_ind]
        neg_negative_ind = np.random.randint(0, self.X.shape[0])
        while neg_negative_ind in neighbour_list_neg:
            neg_negative_ind = np.random.randint(0, self.X.shape[0])

        quadruplets += [self.X[row_index], self.X[neighbour_ind],
                     self.X[negative_ind], self.X[neg_negative_ind]]
        return quadruplets



class LabeledAnnoyTripletGenerator(Sequence):

    def __init__(self, X, Y, annoy_index, k=150, batch_size=32, search_k=-1):
        self.X, self.Y = X, Y
        self.annoy_index = annoy_index
        self.k = k
        self.batch_size = batch_size
        self.search_k = search_k

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size,
                              min((idx + 1) * self.batch_size,
                                  self.X.shape[0]))

        label_batch = self.Y[batch_indices]
        triplet_batch = [self.knn_triplet_from_annoy_index(row_index)
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return ([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]],
                [np.array(label_batch), np.array(label_batch)])

    def knn_triplet_from_annoy_index(self, row_index):
        """ A random (unweighted) positive example chosen. """
        triplet = []
        neighbour_list = np.array(self.annoy_index.get_nns_by_item(row_index, self.k + 1,
                                  search_k=self.search_k, include_distances=False), dtype=np.uint32)

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplet += [self.X[row_index], self.X[neighbour_ind], self.X[negative_ind]]
        return triplet


class LabeledKnnTripletGenerator(Sequence):

    def __init__(self, X, Y, neighbour_matrix, batch_size=32):
        self.X, self.Y = X, Y
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0]))

        label_batch = self.Y[batch_indices]
        triplet_batch = [self.knn_triplet_from_neighbour_list(row_index, self.neighbour_matrix[row_index])
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return ([triplet_batch[:, 0],
                 triplet_batch[:, 1],
                 triplet_batch[:, 2]],
                [np.array(label_batch), np.array(label_batch)])

    def knn_triplet_from_neighbour_list(self, row_index, neighbour_list):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplets += [self.X[row_index],
                     self.X[neighbour_ind],
                     self.X[negative_ind]]
        return triplets
