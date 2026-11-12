
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Simulated data for demonstration purposes
data = pd.DataFrame({
    "Region": [
        "Northeast", "Southeast", "Midwest",
        "Southwest", "West"
    ],
    "Hospitalizations per 10,000": [42, 68, 51, 59, 37],
    "Population": [1200000, 1500000, 950000, 800000, 1100000]
})

st.title("Healthcare Analytics Dashboard")
st.caption("Demonstration using simulated data — not actual healthcare statistics.")

# Summary metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Average hospitalization rate",
    f"{data['Hospitalizations per 10,000'].mean():.1f}"
)

col2.metric(
    "Highest-rate region",
    data.loc[
        data["Hospitalizations per 10,000"].idxmax(),
        "Region"
    ]
)

col3.metric(
    "Regions analyzed",
    len(data)
)

st.divider()

# Interactive chart
st.subheader("Preventable Hospitalization Rates by Region")

fig = px.bar(
    data,
    x="Region",
    y="Hospitalizations per 10,000",
    color="Hospitalizations per 10,000",
    color_continuous_scale="Blues",
    labels={
        "Hospitalizations per 10,000": "Rate per 10,000"
    },
    title="Simulated regional hospitalization rates"
)

st.plotly_chart(fig, use_container_width=True)

# Data table
st.subheader("Underlying Data")
st.dataframe(data, use_container_width=True)
