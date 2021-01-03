from scipy import sparse
import numpy as np
from glcoarsener import Coarsener as GLCoarsener
import math


class Coarsener:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.number_of_nodes = adjacency_matrix.shape[0]

    def gl_coarsener(self):
        coarsener = GLCoarsener(self.adjacency_matrix)
        prolongation_operator = coarsener.apply(
            dimensions=100,
            walk_length=20,
            num_walks=10,
            p=0.1,
            q=1,
            number_of_clusters=self.adjacency_matrix.shape[0] // 5,
            clustering_method='kmeans',
            workers=8
        )
        return prolongation_operator

    def beck(self):
        """
        We sequentially iterate over all nodes. If selected node
        is not in coarse set or fine set, we append the node to
        the coarse set.
        Then we append all the unlabeled connected nodes to fine set.
        """
        fine = []
        coarse = []
        for node in range(self.number_of_nodes):
            if node % 100 == 0:
                print(
                    f'\rCoarse/Fine split: {(node / self.number_of_nodes) * 100}%', end='')
            if node not in fine and node not in coarse:
                # Append to coarse set
                coarse.append(node)
                # Find the connected nodes to this node
                # Connected nodes in row of adj. matrix
                connected_nodes_in_row = set(np.argwhere(
                    self.adjacency_matrix[node] != 0)[:, 1])  # - set([node])
                # Connected nodes in column of adj. matrix
                connected_nodes_in_column = set(np.argwhere(
                    self.adjacency_matrix[:, node] != 0)[:, 0])  # - set([node])
                # All connected nodes
                connected_nodes = connected_nodes_in_column | connected_nodes_in_row
                # We assign all the unlabled connected nodes to fine set
                for neighbor_node in connected_nodes:
                    if neighbor_node not in fine and neighbor_node not in coarse:
                        fine.append(neighbor_node)
        print('\n', end='')
        p = sparse.lil_matrix((self.adjacency_matrix.shape[0], len(coarse)))
        for node in coarse:
            # Identity for coarse nodes in prolongation matrix
            p[node, np.argwhere(np.array(coarse) == node).reshape(-1)] = 1

        for node in fine:
            if np.argwhere(fine == node) % 100 == 0:
                print(
                    f'\rProlongator: {100 * (np.argwhere(fine==node)[0][0] / len(fine))}%', end='')
            # Find all connected nodes to the current fine node
            # Connected nodes in row of adj. matrix
            connected_nodes_in_row = set(np.argwhere(
                self.adjacency_matrix[node] != 0)[:, 1]) - set([node])
            # Connected nodes in column of adj. matrix
            connected_nodes_in_column = set(np.argwhere(
                self.adjacency_matrix[:, node] != 0)[:, 0]) - set([node])
            # All connected nodes
            connected_nodes = connected_nodes_in_column | connected_nodes_in_row
            # From all the connected nodes, find coarse nodes
            connected_coarse = connected_nodes & set(coarse)
            # Calculate sigma (number of coarse nodes in all connected nodes)
            sigma = len(connected_coarse)
            # Find index of each connected coarse node in the list of all coarse nodes
            connected_coarse = list(map(
                lambda con_coarse: True if con_coarse in connected_coarse else False, coarse))
            connected_coarse = [coarse_number for coarse_number, is_connected in enumerate(
                connected_coarse) if is_connected]
            # According to back algorithm, fill the prolongator matrix
            p[node, connected_coarse] = 1 / sigma
        print('\n', end='')
        return p.tocsr()

    def standard_aggregation(self, epsilon):
        def is_strong(i, j):
            tmp = math.sqrt(
                abs(self.adjacency_matrix[i, i] * self.adjacency_matrix[j, j]))
            return abs(self.adjacency_matrix[i, j]) >= epsilon * tmp

        # Find strongly coupled neighborhood to each node
        strongly_coupled_neighborhoods = [[]
                                          for _ in range(self.number_of_nodes)]
        for node in range(self.number_of_nodes):
            # Find the connected nodes to this node
            # Connected nodes in row of adj. matrix
            connected_nodes_in_row = set(np.argwhere(
                self.adjacency_matrix[node] != 0)[:, 1])
            # Connected nodes in column of adj. matrix
            connected_nodes_in_column = set(np.argwhere(
                self.adjacency_matrix[:, node] != 0)[:, 0])
            # All connected nodes
            connected_nodes = connected_nodes_in_column | connected_nodes_in_row
            for connected_node in connected_nodes:
                if is_strong(node, connected_node):
                    strongly_coupled_neighborhoods[node].append(connected_node)
        R = []
        groups = []
        for node in range(self.number_of_nodes):
            if len(strongly_coupled_neighborhoods[node]) == 1:
                groups.append([node])
            else:
                R.append(node)
        # First phase
        R_copy = R[:]
        for i in R_copy:
            if np.argwhere(np.array(R_copy) == i)[0][0] % 100 == 0:
                print(
                    f'\rFirst Phase: {(np.argwhere(np.array(R_copy) == i)[0][0] / len(R_copy)) * 100}%', end='')
            if set(strongly_coupled_neighborhoods[i]).issubset(set(R)):
                groups.append(strongly_coupled_neighborhoods[i])
                R = list(set(R) - set(strongly_coupled_neighborhoods[i]))
        print('\n', end='')
        # Second phase
        R_copy = R
        for i in R_copy:
            if np.argwhere(np.array(R_copy) == i)[0][0] % 100 == 0:
                print(
                    f'\rSecond Phase: {(np.argwhere(np.array(R_copy) == i)[0][0] / len(R_copy)) * 100}%', end='')
            group_to_add = -1
            mutual_strong_connections = -1
            for index, group in enumerate(groups):
                if set(strongly_coupled_neighborhoods[i]).intersection(set(group)):
                    if mutual_strong_connections < len(set(strongly_coupled_neighborhoods[i]).intersection(set(group))):
                        group_to_add = index
                        mutual_strong_connections = len(
                            set(strongly_coupled_neighborhoods[i]).intersection(set(group)))
            if group_to_add > -1:
                groups[group_to_add].append(i)
                R = list(set(R) - set([i]))
        print('\n', end='')
        p = sparse.lil_matrix((self.adjacency_matrix.shape[0], len(groups)))

        def find_cluster_of_node(node_id):
            for index, cluster in enumerate(groups):
                if node_id in cluster:
                    return index
        for node in range(self.adjacency_matrix.shape[0]):
            if node % 100 == 0:
                print(
                    f'\rProlongator: {(node / self.adjacency_matrix.shape[0]) * 100}%', end='')
            p[node, find_cluster_of_node(node)] = 1
        print('\n', end='')
        return p.tocsr()
