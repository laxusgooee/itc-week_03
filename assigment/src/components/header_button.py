import streamlit as st

# Custom CSS to make the button transparent and overlay the header
st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] div:has(div.overlay-button) {
        position: relative;
    }
    .overlay-button button {
        position: absolute;
        width: 100%;
        height: 100%;
        opacity: 0;
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True)

def header_button(label, key):
    container = st.container()
    with container:
        # The invisible button
        st.markdown('<div class="overlay-button">', unsafe_allow_html=True)
        if st.button("", key=key):
            st.session_state[f"{key}_clicked"] = True
        st.markdown('</div>', unsafe_allow_html=True)
        
        # The visible header
        st.header(label)