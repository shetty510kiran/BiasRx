
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="BiasRx", layout="wide")

st.title("🧠 BiasRx - Your Conversational Data Therapist")
st.markdown("Upload a dataset and ask: *“Is this dataset fair?”*")

# Sidebar walkthrough
with st.sidebar:
    st.header("🧭 Walkthrough")
    st.markdown("""
    **Step 1:** Upload your CSV file  
    **Step 2:** View dataset preview & types  
    **Step 3:** Explore patterns and missing values  
    **Step 4:** Detect label/class imbalance  
    **Step 5:** Read BiasRx's insights  
    """)

# Safe display utility
def safe_display_dataframe(df: pd.DataFrame, caption="🧾 Preview of DataFrame"):
    try:
        st.write(caption)
        st.dataframe(df)
    except Exception as e:
        st.warning(f"⚠️ Serialization error: {e}")
        df_safe = df.copy()
        object_cols = df_safe.select_dtypes(include="object").columns
        df_safe[object_cols] = df_safe[object_cols].astype("string")
        st.info("🔧 Converted object-type columns to string to fix issue.")
        st.dataframe(df_safe)

# Upload
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean for safe display
    df_clean = df.copy()
    object_cols = df_clean.select_dtypes(include="object").columns
    df_clean[object_cols] = df_clean[object_cols].astype("string")
    st.success("✅ Dataset uploaded and cleaned!")

    # Tabs for EDA and Bias Detection
    tab1, tab2 = st.tabs(["📊 Exploratory Analysis", "🎯 Bias Detection"])

    with tab1:
        st.subheader("📊 Exploratory Data Analysis")

        st.write("Shape:", df_clean.shape)
        safe_display_dataframe(df_clean.head(), "🧾 First 5 Rows")

        st.write("📘 Column Types:")
        st.dataframe(df_clean.dtypes)

        st.write("🔍 Missing Values:")
        st.dataframe(df_clean.isnull().sum())

        st.write("📈 Unique Values per Column:")
        unique_counts = df_clean.nunique().sort_values(ascending=False)
        st.dataframe(unique_counts)

        st.write("📌 Column Selector")
        selected_col = st.selectbox("View distribution of column:", df_clean.columns)
        chart_data = df_clean[selected_col].value_counts()
        st.bar_chart(chart_data)

        # Export chart data
        st.download_button("📥 Download Chart Data", chart_data.to_csv().encode(), file_name=f"{selected_col}_distribution.csv")

    with tab2:
        st.subheader("🎯 Label Bias Detection")

        possible_targets = [col for col in df_clean.columns if df_clean[col].nunique() < 50]
        target_column = st.selectbox("Select your label column:", options=possible_targets)

        if target_column:
            label_counts = df_clean[target_column].value_counts()
            st.write(f"Distribution of `{target_column}`:")
            st.bar_chart(label_counts)

            st.download_button("📥 Download Label Distribution", label_counts.to_csv().encode(), file_name=f"{target_column}_label_counts.csv")

            max_count = label_counts.max()
            min_count = label_counts.min()

            if max_count > 0:
                imbalance_ratio = min_count / max_count

                if imbalance_ratio < 0.5:
                    st.warning(f"⚠️ Imbalance detected! Smallest class is only {imbalance_ratio*100:.1f}% of the largest.")
                else:
                    st.success("✅ Dataset looks balanced.")

                st.subheader("🛠️ Bias Fix Suggestions")
                if imbalance_ratio < 0.5:
                    st.markdown("""
                    Try one of these:
                    - **SMOTE:** Create synthetic examples for the minority class  
                    - **Oversampling:** Duplicate underrepresented class rows  
                    - **Undersampling:** Reduce majority class instances  
                    - **Class weighting:** Adjust model training sensitivity  
                    """)

                # 💬 Chat-style insight from BiasRx
                st.subheader("🤖 BiasRx Insight")
                if st.button("🧠 Ask BiasRx: Why is this biased?"):
                    insight = f"""Hello! 👋 I'm BiasRx.

Based on the distribution of `{target_column}`, I noticed that one class is significantly underrepresented.

This could make your model biased by:
- Learning patterns only from the majority class
- Failing to recognize edge cases
- Creating unfair predictions for the underrepresented group

Consider balancing this before training! 💡"""
                    st.info(insight)
