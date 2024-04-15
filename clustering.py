import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from utils import Utils

class CustomerSegmentation:
    COLOR_SCHEME = "Plasma"

    @staticmethod
    def do_segment(K, df):
        st.write(
            f"Your model is <b style='color: green'>Kmean Clustering</b> with <b style='color: green'>{K}</b> clusters", unsafe_allow_html=True)
        kmean = KMeans(n_clusters=K, init='random', random_state=42, n_init=10)
        return kmean.fit(df)

    # Create the K means model for different values of K
    @staticmethod
    def try_different_clusters(K, data):

        cluster_values = list(range(1, K+1))
        inertias = []

        for c in cluster_values:
            model = KMeans(n_clusters=c, init='random', random_state=12, n_init=10)
            model.fit(data)
            inertias.append(model.inertia_)

        distances = pd.DataFrame(
            {"clusters": list(range(1, K+1)), "sum of squared distances": inertias})
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=distances["clusters"], y=distances["sum of squared distances"]))

        figure.update_layout(xaxis=dict(tick0=1, dtick=1, tickmode='linear'),
                             xaxis_title="Number of clusters",
                             yaxis_title="Sum of squared distances",
                             title_text="Finding optimal number of segmentations using elbow method")
        st.plotly_chart(figure, theme="streamlit")
        st.markdown("<b>💡Hint: Choose the number of segmentation at the elbow point (e.g., if the plot looks like an elbow at K=3, then use 3 clusters). </b>", unsafe_allow_html=True)

    @staticmethod
    @st.cache_data
    def visualize_cluster_scatter3d(df, segmentation_col, df_cols):
        fig = px.scatter_3d(
            df,
            color=segmentation_col,
            x=df_cols[0],
            y=df_cols[1],
            z=df_cols[2],
            color_continuous_scale=CustomerSegmentation.COLOR_SCHEME,
            title="Segmentations in 3D"
        )
        fig.update_layout(height=600, dragmode=False)
        # fig.update_traces(marker_size = 2)
        st.plotly_chart(fig)

    @staticmethod
    @st.cache_data
    def visualize_cluster_bar(df, x, segmentation_col):
        grouped_df = df.groupby([x, segmentation_col], as_index=False).size()
        fig = px.bar(grouped_df, x=x, y="size", color=segmentation_col,
                     title=f"Segmentations distribution by {x}", color_continuous_scale=CustomerSegmentation.COLOR_SCHEME)
        st.plotly_chart(fig)

    @staticmethod
    @st.cache_data
    def visualize_cluster_hist(df, x, k, segmentation_col):
        n_colors = k
        color_scale = px.colors.sample_colorscale(CustomerSegmentation.COLOR_SCHEME, [
                                                  n/(n_colors - 1) for n in reversed(range(n_colors))])
        fig = px.histogram(df, x=x, nbins=10, color=segmentation_col,
                           color_discrete_sequence=color_scale, title=f"Segmentations distribution by {x}")
        st.plotly_chart(fig)


    @staticmethod
    def use_model(df):
        # Split the columns into 3 equal parts
        list_columns = df.columns.values.tolist()
        df_cols = Utils.split_list(list_columns, 3)

        # Create 3 columns for input widgets
        col1, col2, col3 = st.columns(3)
        values = []
        # Iterate over the columns in each part
        for i, cols in enumerate([df_cols[0], df_cols[1], df_cols[2]]):
            with [col1, col2, col3][i]:
                for col in cols:
                    if df[col].dtypes == 'object':
                        value = st.selectbox(col, options=df[col].unique())
                    else:
                        value = st.number_input(col, value=df[col].mean())
                    values.append(value)

        df_test = pd.DataFrame(data=[values], columns=list_columns)
        return df_test
