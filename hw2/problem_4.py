import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from utils import KdTree


def generate_S_data(n, r):
    sampled_t = np.random.uniform(0, 2 * np.pi, n)
    x = r * np.cos(sampled_t)
    y = r * np.sin(sampled_t)
    data = np.stack([x, y], axis=-1)
    return data


def generate_H_data(n):
    sampled_t = np.random.uniform(0, 2 * np.pi, n)
    x = np.cos(sampled_t)
    y = np.sin(sampled_t)
    data = np.stack([x, y, sampled_t], axis=-1)
    return data


def subspace_fitting(data, n=1):
    within = 1

    A_list = []
    for target_x in data:
        tree = KdTree(data)
        data_within_distance = tree.search(target=target_x, within=within)
        Y = data_within_distance - target_x

        def find_y(Y, y_idx_list, criteria_list):
            inner_max_y = np.max(criteria_list, axis=0)
            # A mask to exclude chosen y's in previous iterations
            inner_max_y[y_idx_list] = np.inf
            y_idx = np.argmin(inner_max_y)
            return y_idx, Y[y_idx]

        y_list = []
        y_idx_list = []
        criteria_list = [np.abs(within - np.linalg.norm(Y, axis=1))]
        for _ in range(n):
            y_idx, y = find_y(Y, y_idx_list, criteria_list)
            y_idx_list.append(y_idx)
            y_list.append(y)
            criteria_list.append(np.abs(np.inner(y / np.linalg.norm(y), Y)))

        y_list = np.array(y_list)

        # Span of A is a line
        if len(y_list) > 1:
            A_list.append([target_x, target_x + y_list])
        else:
            A_list.append([target_x, target_x + y_list[0]])

    return np.array(A_list)


np.random.seed(2023)
# Generate the sphere data
data = generate_S_data(100, 5)
data_df = pd.DataFrame(data)
fig = px.scatter(x=data_df[0], y=data_df[1], hover_data=[data_df.index])
fig.update_xaxes(range=[-6, 6])
fig.update_yaxes(range=[-6, 6])
fig.update_layout(title=dict(text="Subset of S", x=0.5,
                  xanchor="center"), showlegend=False)
fig.update_layout(width=800, height=800)
fig.write_html("./figures/S_samples.html")

result = subspace_fitting(data)
label_list = np.arange(0, 100)
result_dict = {"label": np.repeat(label_list, 2, axis=-1),
               "x": result[:, :, 0].flatten(),
               "y": result[:, :, 1].flatten()}
result_df = pd.DataFrame(result_dict)
fig = px.line(result_df, x="x", y="y",
              hover_data=["label"], color="label",
              color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_xaxes(range=[-6, 6])
fig.update_yaxes(range=[-6, 6])
fig.update_layout(title=dict(text="Subspace Fitting of S",
                  x=0.5, xanchor="center"), showlegend=False)
fig.update_layout(width=800, height=800)
fig.write_html("./figures/S_subspace_fitting.html")


# Generate the helical data
data = generate_H_data(100)
data_df = pd.DataFrame(data)
fig = px.scatter_3d(x=data_df[0], y=data_df[1],
                    z=data_df[2], hover_data=[data_df.index])
fig.update_traces(marker=dict(size=3))
fig.update_layout(title=dict(text="Subset of H", x=0.5,
                  xanchor="center"), showlegend=False)
fig.update_layout(width=800, height=800)
fig.write_html("./figures/H_samples.html")

result = subspace_fitting(data)
label_list = np.arange(0, 100)
result_dict = {"label": np.repeat(label_list, 2, axis=-1),
               "x": result[:, :, 0].flatten(),
               "y": result[:, :, 1].flatten(),
               "z": result[:, :, 2].flatten()}
result_df = pd.DataFrame(result_dict)
fig = px.line_3d(result_df, x="x", y="y", z="z",
                 hover_data=["label"], color="label",
                 color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_layout(title=dict(text="Subspace Fitting of H",
                  x=0.5, xanchor="center"), showlegend=False)
fig.update_layout(width=800, height=800)
fig.write_html("./figures/H_subspace_fitting.html")
