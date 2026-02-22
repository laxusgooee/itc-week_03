import streamlit as st

st.title("Business Insights")
st.markdown("""
## Overview

Using K-Means clustering, 9 distinct player segments were identified.  
The optimal number of clusters (k = 9) was determined using the Silhouette Score, which peaked at 0.28, indicating moderate but meaningful cluster separation.

Although nine statistical clusters were detected, they can be strategically grouped into three major business archetypes.
""")

with st.expander("Segment 1: Elite Young Assets (Cluster 7)"):
    st.markdown("""
        
### Segment 1: Elite Young Assets (Cluster 7)

**Profile:**
- Youngest average age (~23 years)
- Highest overall rating
- Highest potential rating
- Highest market value (~€108M)
- Slightly lower minutes and goal output compared to veterans

**Business Interpretation:**
These players represent long-term strategic investments.  
They combine strong current performance with high development potential.

**Strategic Actions:**
- Secure long-term contracts
- Build squad structure around them
- Avoid selling unless receiving premium transfer offers

This cluster represents the club’s future growth engine.
    """)

with st.expander("Segment 2: Veteran High-Output Performers (Cluster 5)"):
    st.markdown("""
        
## Segment 2: Veteran High-Output Performers (Cluster 5)

**Profile:**
- Oldest average age (~32 years)
- Highest goals and assists
- Highest minutes played
- Lower potential rating
- Lower market value relative to output

**Business Interpretation:**
These players are short-term performance maximizers.  
They provide immediate competitive advantage but carry age-related decline risk.

**Strategic Actions:**
- Utilize for win-now strategies
- Avoid long-term financial commitments
- Consider selling before performance decline

This cluster represents short-term competitive assets.

    """)

with st.expander("Segment 3: Prime Stable Performers (Clusters 0,1,2,3,4,6,8)"):
    st.markdown("""
        

## Segment 3: Prime Stable Performers (Clusters 0,1,2,3,4,6,8)

**Profile:**
- Average age ~27–28 (prime years)
- Balanced overall and potential ratings
- Consistent match participation
- Stable market valuation (~€88–93M)
- Balanced goal and assist contribution

**Business Interpretation:**
These players form the operational backbone of the squad.  
They provide stability, reliability, and medium-term value retention.

**Strategic Actions:**
- Maintain balanced contract durations
- Use as core rotation players
- Monitor for either breakout potential or decline signals

This segment represents the structural foundation of the team.
    """)

st.space()

st.subheader("Strategic Conclusion")

st.markdown("""
While the model identified nine statistically optimal clusters, the player market naturally consolidates into three strategic archetypes:

1. Elite Young Investments  
2. Veteran High-Impact Contributors  
3. Prime-Age Stable Performers  

This segmentation supports:
- Data-driven recruitment strategy  
- Contract optimization  
- Risk management planning  
- Squad lifecycle management  

The clustering model provides a structured framework for strategic football asset management.
""")