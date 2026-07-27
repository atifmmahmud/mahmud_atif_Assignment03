import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("Assignment 5: Part B")
st.write("Submission for Atif M. Mahmud")
st.write("SFU ID: atifm@sfu.ca")

# Load, clean, and display dataframe
with st.expander("Initial dataframe"):
    df = pd.read_pickle("df_businesses.pkl")
    df = df.loc[~df["localarea"].isna()]
    st.subheader("The initial dataframe")
    st.dataframe(df)

## Keep 1000 as a threshold. Drop areas with less than 1000 businesses.
with st.expander("Filtering to >= 1000"):
    businesses_count = df["localarea"].value_counts()
    df = df[df["localarea"].map(businesses_count) >= 1000]
    st.subheader("Value counts of locations, filtered to > 1000")
    st.write(df["localarea"].value_counts())

# From Hints
with st.expander("Cross tab"):
    crosstab_df = pd.crosstab(df["localarea"], df["businesstype"], normalize="index") * 100
    st.subheader("Cross tab showing proportion of business of that type in that")
    st.dataframe(crosstab_df)

# K-means
k = st.sidebar.slider("Number of clusters", 5, 15, 1)
X = crosstab_df.to_numpy()
model = KMeans(n_clusters=k)
labels = model.fit_predict(X)
crosstab_df["cluster_B"] = pd.Categorical(labels.astype(str))
n_clusters_found = crosstab_df["cluster_B"].nunique()
st.metric("Number of clusters:", n_clusters_found)
with st.expander("Dataframe with cluster"):
    st.write(crosstab_df)

# PCA
pca_size = PCA(n_components=2)
X_pca = pca_size.fit_transform(X)
crosstab_df["dim_1"] = X_pca[:,0]
crosstab_df["dim_2"] = X_pca[:,1]

fig_pca = px.scatter(
    crosstab_df,
    x="dim_1",
    y="dim_2",
    color="cluster_B"
)
st.subheader("PCA visualization")
st.plotly_chart(fig_pca, width="stretch")

# Taking average lat/lon to find centroid. Not the best way geographically but good enough for now.
# Also taking count of total num businesses
with st.expander("Intermediate calculations"):
    avg_lat = df.groupby("localarea")["lat"].mean()
    avg_lon = df.groupby("localarea")["lon"].mean()
    st.write("Average/Centroid latitude")
    st.write(avg_lat)
    st.write("Average/Centroid longitude")
    st.write(avg_lon)

# Populate crosstab dataframe with data on lat/lon and total num businesses
crosstab_df["lat"] = avg_lat
crosstab_df["lon"] = avg_lon
crosstab_df["num_biz"] = businesses_count
with st.expander("Crosstab DF with all data I need"):
    st.write(crosstab_df)

# Visualize on map
st.subheader("Map visualization")
fig = px.scatter_map(crosstab_df, lat="lat", lon="lon", zoom=10, height=550, map_style="carto-darkmatter", color="cluster_B", size="num_biz")
st.plotly_chart(fig, width="stretch")

# See clustering
with st.expander("Neighborhood and cluster"):
    area_clusters = crosstab_df[["cluster_B"]]
    st.write(area_clusters)