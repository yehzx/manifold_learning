import random
import time

import numpy as np

N = [100, 500, 1000]
K = [5, 10, 50, 100, 500]


class Node():
    def __init__(self, id_):
        self.id = id_
        self.parent = self
        self.children = []
        self.n_children = 0

    def become_child(self, parent):
        self.parent = parent
        for child in self.children:
            child.become_child(parent=parent)
        self.children = []
        self.n_children = 0

    def become_parent(self, child):
        self.children.append(child)
        self.children.extend(child.children)
        self.n_children = len(self.children)


class NodeCollection():
    def __init__(self, node_list):
        self.node_dict = {node.id: node for node in node_list}
        self.n_nodes = len(self.node_dict)


def main():
    total_start = time.perf_counter()
    for k in K:
        for n in N:
            if k >= n:
                continue
            start = time.perf_counter()

            step_count_list = [single_run(k, n) for i in range(10000)]

            end = time.perf_counter()
            print(f"k = {k}, n = {n}, avg_step_count = {np.mean(step_count_list):.2f}, stdev = {np.std(step_count_list):.2f}")
            print(f"Elapsed time: {end - start:.2f}s")
    total_end = time.perf_counter()
    print(f"Total elapsed time: {total_end - total_start:.2f}s")


def single_run(k, n):
    step_count = 0
    node_collection = initialize_nodes(n)

    while next_step(k, n, node_collection):
        step_count += 1
        assert step_count < 10000, "Loop may be infinite!"

    return step_count


def initialize_nodes(n):
    node_list = [Node(i) for i in range(n)]
    node_collection = NodeCollection(node_list)

    return node_collection


def next_step(k, n, node_collection: NodeCollection):
    nodes_to_connect = sample_k_nodes_from_n(k, n, node_collection)
    parent_node = connect_nodes(nodes_to_connect)
    # If parent_node.n_children == (n - 1), it means that only one component left.
    return False if parent_node.n_children == (n - 1) else True


def sample_k_nodes_from_n(k, n, node_collection):
    k_numbers = random.sample(range(n), k)
    return [node_collection.node_dict[number] for number in k_numbers]


def connect_nodes(nodes_to_connect: list[Node]):
    n_children_of_parent_node = [node.parent.n_children for node in nodes_to_connect]
    parent_node_with_max_children = nodes_to_connect[np.argmax(n_children_of_parent_node)].parent
    for node in nodes_to_connect:
        if node.parent is not parent_node_with_max_children:
            parent_node_with_max_children.become_parent(child=node.parent)
            node.parent.become_child(parent=parent_node_with_max_children)

    return parent_node_with_max_children


if __name__ == "__main__":
    main()


"""
# 2023/09/18
k = 5, n = 100, avg_step_count = 101.01, stdev = 24.53
Elapsed time: 5.86s
k = 5, n = 500, avg_step_count = 674.48, stdev = 125.42
Elapsed time: 40.33s
k = 5, n = 1000, avg_step_count = 1492.21, stdev = 255.35
Elapsed time: 88.78s
k = 10, n = 100, avg_step_count = 48.82, stdev = 11.92
Elapsed time: 4.18s
k = 10, n = 500, avg_step_count = 335.23, stdev = 62.53
Elapsed time: 29.26s
k = 10, n = 1000, avg_step_count = 746.75, stdev = 129.09
Elapsed time: 63.70s
k = 50, n = 100, avg_step_count = 7.02, stdev = 1.83
Elapsed time: 2.08s
k = 50, n = 500, avg_step_count = 64.29, stdev = 12.44
Elapsed time: 18.25s
k = 50, n = 1000, avg_step_count = 145.37, stdev = 24.93
Elapsed time: 39.94s
k = 100, n = 500, avg_step_count = 30.07, stdev = 5.75
Elapsed time: 16.06s
k = 100, n = 1000, avg_step_count = 70.77, stdev = 12.14
Elapsed time: 38.50s
k = 500, n = 1000, avg_step_count = 10.30, stdev = 1.86
Elapsed time: 28.43s
Total elapsed time: 375.37s
"""
