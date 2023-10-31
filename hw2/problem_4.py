import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def subspace_fitting(data):
    within = 1

    A_list = []
    for target_x in data:
        tree = KdTree(data)
        data_within_distance = tree.search(target=target_x, within=within)
        Y = data_within_distance - target_x

        y_idx = np.argmin(np.abs(within - np.linalg.norm(Y)))
        y = Y[y_idx]

        # Span of A is a line 
        A_list.append([target_x + 10 * y, target_x - 10 * y])

    return np.array(A_list)
    

# Generate the sphere data
data = generate_S_data(100, 5)
data_df = pd.DataFrame(data)
fig = px.scatter(x=data_df[0], y=data_df[1], hover_data=[data_df.index])
fig.update_xaxes(range=[-6, 6])
fig.update_yaxes(range=[-6, 6])
fig.update_layout(title=dict(text="Subset of S", x=0.5, xanchor="center"), showlegend=False)
fig.update_layout(width=800, height=800)
fig.write_html("./figures/S_samples.html")


result = subspace_fitting(data)
label_list = np.arange(0, 100)
result_dict = {"label": np.repeat(label_list, 2, axis=-1),
               "x": result[:, :, 0].flatten(),
               "y": result[:, :, 1].flatten()}
result_df = pd.DataFrame(result_dict)
fig = px.line(result_df, x="x", y="y", hover_data=["label"], color="label", color_discrete_sequence= px.colors.sequential.Viridis)
fig.update_xaxes(range=[-6, 6])
fig.update_yaxes(range=[-6, 6])
fig.update_layout(title=dict(text="Subspace Fitting of S", x=0.5, xanchor="center"), showlegend=False)
fig.update_layout(width=800, height=800)
fig.write_html("./figures/S_subspace_fitting.html")


# Generate the helical data
data = generate_H_data(100)
data_df = pd.DataFrame(data)
fig = px.scatter_3d(x=data_df[0], y=data_df[1], z=data_df[2], hover_data=[data_df.index])
fig.update_traces(marker=dict(size=3))
# fig.update_xaxes(range=[-5, 5])
# fig.update_yaxes(range=[-5, 5])
fig.update_layout(title=dict(text="Subset of H", x=0.5, xanchor="center"), showlegend=False)
fig.update_layout(width=800, height=800)
fig.write_html("./figures/H_samples.html")

result = subspace_fitting(data)
label_list = np.arange(0, 100)
result_dict = {"label": np.repeat(label_list, 2, axis=-1),
               "x": result[:, :, 0].flatten(),
               "y": result[:, :, 1].flatten(),
               "z": result[:, :, 2].flatten()}
result_df = pd.DataFrame(result_dict)
fig = px.line_3d(result_df, x="x", y="y", z="z", hover_data=["label"], color="label", color_discrete_sequence= px.colors.sequential.Viridis)
fig.update_layout(title=dict(text="Subspace Fitting of H", x=0.5, xanchor="center"), showlegend=False)
fig.update_layout(width=800, height=800)
fig.write_html("./figures/H_subspace_fitting.html")