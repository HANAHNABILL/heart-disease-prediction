import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")
st.title("🚀 Test Deployment")
st.success("If you see this, basic deployment works!")


try:
    import pandas as pd
    import joblib
    st.info("✅ All packages imported successfully")
except Exception as e:
    st.error(f"Package import error: {e}")
