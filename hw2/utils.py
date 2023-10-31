import numpy as np


class Node():
    def __init__(self, data_point, axis):
        self.coordinates = data_point
        self.axis = axis
        self.left = None
        self.right = None


class KdTree():
    def __init__(self, data):
        self.data = np.array(data)
        self.dim = np.shape(self.data)[1]
        self.root = self.build_tree(self.data)

    def build_tree(self, data: np.ndarray):
        if len(data) == 0:
            return None
        elif len(data) == 1:
            return Node(data_point=data[0], axis=None)

        data_size, data_dim = np.shape(data)
        max_var_axis = np.argmax([np.var(data[:, dim])
                                 for dim in range(data_dim)])
        data = data[np.argsort(data[:, max_var_axis])]

        mid = data_size // 2

        node = Node(data_point=data[mid], axis=max_var_axis)

        in_left = (data[:, max_var_axis] <= data[mid][max_var_axis])
        in_right = ~in_left
        in_left[mid] = False
        in_right[mid] = False

        node.left = self.build_tree(data[in_left])
        node.right = self.build_tree(data[in_right])

        return node

    def search(self, target, within):
        points_within_range = []

        def search_tree(current_node):
            node_list = []
            while current_node is not None:
                node_list.append(current_node)

                if current_node.axis is None:
                    break

                if target[current_node.axis] < current_node.coordinates[current_node.axis]:
                    current_node = current_node.left
                else:
                    current_node = current_node.right

            return node_list

        def find_points_within_range(node_list):
            nonlocal points_within_range
            while node_list:
                back_node: Node
                back_node = node_list.pop()

                if np.linalg.norm(target - back_node.coordinates, ord=2) < within:
                    points_within_range.append(back_node.coordinates)

                if back_node.axis is None:
                    continue

                distance_to_hyperplane = target[back_node.axis] - \
                    back_node.coordinates[back_node.axis]
                if abs(distance_to_hyperplane) < within:
                    if distance_to_hyperplane <= 0:
                        find_points_within_range(search_tree(back_node.right))
                    else:
                        find_points_within_range(search_tree(back_node.left))

        find_points_within_range(search_tree(self.root))

        return points_within_range