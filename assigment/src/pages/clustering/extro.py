import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pages.clustering.data_summary import df_no_index
from pages.clustering.data_preparation import df_scaled_encoded

# re-fit k=9 for summary stats
kmeans = KMeans(n_clusters=9, random_state=42, n_init="auto")
kmeans.fit(df_scaled_encoded)
df_ori = df_no_index.copy()
df_ori["Cluster"] = kmeans.labels_

st.title("Business Insights")
st.markdown("""
## Overview

Using **K-Means clustering**, **9 distinct player segments** were identified.  
The optimal k was confirmed by the **Silhouette Score** (peak at k=9) and the **WCSS elbow**.

Although nine statistical clusters were detected, they consolidate into **three strategic archetypes**.
""")

# strategy summary table
summary_df = pd.DataFrame({
    "Archetype": ["Elite Young Assets", "Veteran High-Output", "Prime Stable Performers"],
    "Key Clusters": ["Cluster 7", "Cluster 5", "Clusters 0,1,2,3,4,6,8"],
    "Avg Age": ["~23 yrs", "~32 yrs", "~27–28 yrs"],
    "Avg Rating": ["Highest", "High", "Balanced"],
    "Market Value": ["~€108M", "Lower (vs output)", "~€88–93M"],
    "Strategy": ["Lock in long-term contracts", "Win-now, avoid long commitments", "Core rotation, monitor closely"],
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()

# segment deep-dives
with st.expander("🌟 Segment 1: Elite Young Assets (Cluster 7)"):
    st.markdown("""
**Profile:**
- Youngest average age (~23 years)
- Highest overall & potential rating
- Highest market value (~€108M)
- Slightly lower goal output vs veterans

**Business Interpretation:**  
Long-term strategic investments combining current performance with high development potential.

**Strategic Actions:**
- Secure long-term contracts early
- Build squad structure around them
- Only sell at significant premium prices

> This cluster represents the club's **future growth engine**.
    """)

with st.expander("💪 Segment 2: Veteran High-Output Performers (Cluster 5)"):
    st.markdown("""
**Profile:**
- Oldest average age (~32 years)
- Highest goals and assists
- Highest minutes played
- Lower potential rating, lower market value relative to output

**Business Interpretation:**  
Short-term performance maximizers providing immediate competitive advantage.

**Strategic Actions:**
- Use for win-now strategies
- Avoid long-term financial commitments
- Plan exit before performance decline

> This cluster represents **short-term competitive assets**.
    """)

with st.expander("⚖️ Segment 3: Prime Stable Performers (Clusters 0,1,2,3,4,6,8)"):
    st.markdown("""
**Profile:**
- Average age ~27–28 (prime years)
- Balanced overall and potential ratings
- Consistent match participation
- Stable market valuation (~€88–93M)

**Business Interpretation:**  
Operational backbone — stable, reliable, medium-term value retention.

**Strategic Actions:**
- Maintain balanced contract durations (2–3 years)
- Use as core rotation squad
- Monitor for breakout potential or early decline signals

> This segment represents the **structural foundation** of the team.
    """)

st.divider()

# market value distribution by archetype (simulated from cluster labels)
st.subheader("Market Value by Cluster")
mv_cluster = (
    df_ori.groupby("Cluster")["market_value_million_eur"]
    .mean()
    .reset_index()
    .rename(columns={"market_value_million_eur": "Avg Market Value (€)"})
)
mv_cluster["Cluster"] = mv_cluster["Cluster"].astype(str)

mv_bar = (
    alt.Chart(mv_cluster)
    .mark_bar()
    .encode(
        x=alt.X("Cluster:N", sort="-y"),
        y=alt.Y("Avg Market Value (€):Q", title="Avg Market Value (€)"),
        color=alt.Color("Cluster:N", scale=alt.Scale(scheme="tableau10"), legend=None),
        tooltip=["Cluster:N", alt.Tooltip("Avg Market Value (€):Q", format=",.0f")],
    )
    .properties(title="Average Market Value per Cluster", width="container", height=320)
    .interactive()
)
st.altair_chart(mv_bar, use_container_width=True)

# age vs rating bubble by cluster
st.subheader("Age vs Rating by Cluster")
bubble_df = df_ori.groupby("Cluster").agg(
    AvgAge=("age", "mean"),
    AvgRating=("overall_rating", "mean"),
    AvgMarketValue=("market_value_million_eur", "mean"),
    Count=("age", "count"),
).reset_index()
bubble_df["Cluster"] = bubble_df["Cluster"].astype(str)

bubble = (
    alt.Chart(bubble_df)
    .mark_circle(opacity=0.85)
    .encode(
        x=alt.X("AvgAge:Q", title="Avg Age", scale=alt.Scale(zero=False)),
        y=alt.Y("AvgRating:Q", title="Avg Overall Rating", scale=alt.Scale(zero=False)),
        size=alt.Size("AvgMarketValue:Q", title="Avg Market Value", scale=alt.Scale(range=[200, 2500])),
        color=alt.Color("Cluster:N", scale=alt.Scale(scheme="tableau10"), title="Cluster"),
        tooltip=[
            "Cluster:N",
            alt.Tooltip("AvgAge:Q", format=".1f"),
            alt.Tooltip("AvgRating:Q", format=".1f"),
            alt.Tooltip("AvgMarketValue:Q", format=",.0f"),
            "Count:Q",
        ],
    )
    .properties(title="Cluster Summary: Age vs Rating (bubble = market value)", width="container", height=400)
    .interactive()
)
st.altair_chart(bubble, use_container_width=True)

st.divider()

# strategic conclusion
st.subheader("Strategic Conclusion")
st.markdown("""
| Model Output | Business Value |
|---|---|
| 9 statistical clusters | Identifies subtle sub-groups within the squad |
| 3 strategic archetypes | Actionable categories for contract & transfer decisions |
| Cluster heatmap | Shows which features drive each segment |
| Market value analysis | Quantifies the financial impact of each archetype |

**Recommended Applications:**
1. **Recruitment** — target Elite Young Asset profile players in the transfer market
2. **Contract Management** — match contract length to player archetype
3. **Squad Planning** — ensure a healthy balance across all three archetypes
4. **Performance Monitoring** — track cluster migration year over year (Prime → Veteran)
""")