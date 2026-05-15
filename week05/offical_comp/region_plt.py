import pandas as pd
import plotly.express as px

# Change this to your file
FILE_PATH = "test_cleaned.csv"

# Load data
df = pd.read_csv(FILE_PATH, low_memory=False)

# Keep required columns
df = df[["latitude", "longitude", "label", "name", "city", "stars"]].copy()

# Clean data
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
df = df.dropna(subset=["latitude", "longitude", "label"])

# Optional: keep only top restaurant types if legend gets too crowded
# top_labels = df["label"].value_counts().head(10).index
# df = df[df["label"].isin(top_labels)]

# Build interactive scatter map
fig = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    color="label",
    hover_name="name",
    hover_data={
        "city": True,
        "stars": True,
        "latitude": False,
        "longitude": False,
        "label": True
    },
    zoom=3,
    height=700,
    title="Restaurant Locations by Type"
)

fig.update_traces(marker=dict(size=8, opacity=0.6))
fig.update_layout(
    mapbox_style="open-street-map",
    legend_title="Restaurant Type",
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

fig.show()

# Optional save to HTML
fig.write_html("restaurant_map_by_type.html")
