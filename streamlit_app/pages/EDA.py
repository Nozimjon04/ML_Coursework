import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(page_title="EDA", layout="wide")

st.title("Exploratory Data Analysis")

# Helper: Load dataset
@st.cache_data
def load_csv(uploaded_file):
    # read csv safely
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        # try reading with different encoding
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)

# Prefer cleaned_df from session_state (preprocessing), otherwise allow upload
df = None
if "cleaned_df" in st.session_state:
    df = st.session_state["cleaned_df"]
    st.success("Using preprocessed data from session state.")
else:
    uploaded_file = st.file_uploader("Upload contract awards CSV file (or use Preprocessing page first)", type=["csv"])
    if uploaded_file:
        df = load_csv(uploaded_file)
        st.success("File uploaded and loaded.")

if df is None:
    st.info("No dataset loaded yet. Upload a CSV file or run preprocessing on the Preprocessing page.")
    st.stop()

# Basic cleaning / make a copy
df = df.copy()

# Ensure consistent column names (strip)
df.columns = [c.strip() for c in df.columns]

# Sidebar controls
with st.sidebar:
    st.header("EDA Controls")
    show_head = st.checkbox("Show dataset head", value=True)
    n_head = st.number_input("Rows to show", min_value=3, max_value=50, value=5)
    show_tail = st.checkbox("Show dataset tail", value=False)
    n_tail = st.number_input("Tail rows", min_value=1, max_value=50, value=5)
    show_sample = st.checkbox("Show random sample", value=False)
    sample_n = st.number_input("Sample size", min_value=1, max_value=1000, value=10)
    st.markdown("---")
    st.write("Choose columns for plots")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    st.write(f"Numeric cols: {len(numeric_cols)}")
    st.write(f"Categorical cols: {len(categorical_cols)}")

# Top-level info
st.subheader("Dataset shape & missing values")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.metric("Rows", df.shape[0])
with c2:
    st.metric("Columns", df.shape[1])
with c3:
    st.metric("Duplicate rows", int(df.duplicated().sum()))

st.markdown("**Missing values (per column)**")
missing = df.isnull().sum().sort_values(ascending=False)
st.dataframe(pd.DataFrame({"missing_count": missing, "missing_pct": (missing/len(df)).round(4)}).query("missing_count>0"))

st.markdown("---")

# Statistical summary
st.subheader("Statistical summary for numeric and object columns")
with st.expander("Numeric summary (describe)"):
    st.write(df.describe(include=[np.number]).T.style.format("{:.4g}"))

with st.expander("Categorical summary (top & freq)"):
    cat_summary = []
    for col in categorical_cols:
        top = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
        freq = int(df[col].value_counts(dropna=True).iloc[0]) if df[col].nunique()>0 else 0
        cat_summary.append((col, df[col].nunique(), top, freq))
    cat_df = pd.DataFrame(cat_summary, columns=["column", "n_unique", "top", "top_freq"])
    st.dataframe(cat_df.sort_values("n_unique", ascending=False))

st.markdown("---")

# Correlation matrix (numeric)
st.subheader("Correlation matrix (numeric columns)")
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap (numeric columns)")
    st.pyplot(fig)
else:
    st.warning("Not enough numeric columns to show correlation matrix.")

st.markdown("---")

# Histogram & boxplot controls
st.subheader("Distributions & Outliers")

col1, col2 = st.columns(2)
with col1:
    hist_col = st.selectbox("Select numeric column for histogram", options=numeric_cols, index=0 if numeric_cols else None)
    bins = st.slider("Bins", 5, 200, 30)
    kde = st.checkbox("Show KDE (histogram)", value=True)
    log_hist = st.checkbox("Log scale X (histogram)", value=False)
with col2:
    box_col = st.selectbox("Select numeric column for boxplot", options=numeric_cols, index=1 if len(numeric_cols)>1 else 0)
    show_points = st.checkbox("Show underlying points on boxplot", value=False)

# Histogram
if hist_col:
    fig_h, ax_h = plt.subplots(figsize=(8,4))
    if log_hist:
        sns.histplot(np.log1p(df[hist_col].dropna()), bins=bins, kde=kde, ax=ax_h)
        ax_h.set_xlabel(f"ln(1 + {hist_col})")
    else:
        sns.histplot(df[hist_col].dropna(), bins=bins, kde=kde, ax=ax_h)
        ax_h.set_xlabel(hist_col)
    ax_h.set_ylabel("Count")
    ax_h.set_title(f"Histogram of {hist_col}")
    st.pyplot(fig_h)

# Boxplot
if box_col:
    fig_b, ax_b = plt.subplots(figsize=(8,3))
    sns.boxplot(x=df[box_col].dropna(), ax=ax_b)
    ax_b.set_title(f"Boxplot of {box_col}")
    if show_points:
        sns.stripplot(x=df[box_col].dropna(), color="0.3", alpha=0.3, jitter=0.2, ax=ax_b)
    st.pyplot(fig_b)

st.markdown("---")

# Scatter plot between two numeric columns
st.subheader("Scatter plot (numeric vs numeric)")
if len(numeric_cols) >= 2:
    sp_col_x = st.selectbox("X (actual)", options=numeric_cols, index=0)
    sp_col_y = st.selectbox("Y (predicted / target)", options=numeric_cols, index=1)
    sample_scatter = st.number_input("Scatter sample size (0 = all)", min_value=0, max_value=50000, value=2000)
    df_scatter = df[[sp_col_x, sp_col_y]].dropna()
    if sample_scatter and sample_scatter>0 and len(df_scatter)>sample_scatter:
        df_scatter = df_scatter.sample(sample_scatter, random_state=42)
    fig_s, ax_s = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=df_scatter, x=sp_col_x, y=sp_col_y, alpha=0.6, s=40, ax=ax_s)
    ax_s.set_title(f"Scatter: {sp_col_x} vs {sp_col_y}")
    st.pyplot(fig_s)
else:
    st.info("Need at least two numeric columns for scatter plot.")

st.markdown("---")

# Categorical plots
st.subheader("Categorical distributions (top categories)")
if categorical_cols:
    cat_choice = st.selectbox("Select categorical column", options=categorical_cols)
    top_k = st.slider("Top K categories to show", min_value=3, max_value=50, value=10)
    vc = df[cat_choice].value_counts(dropna=True).head(top_k)
    fig_c, ax_c = plt.subplots(figsize=(8,4))
    sns.barplot(x=vc.values, y=vc.index, ax=ax_c)
    ax_c.set_xlabel("Count")
    ax_c.set_ylabel(cat_choice)
    ax_c.set_title(f"Top {top_k} values in {cat_choice}")
    st.pyplot(fig_c)
else:
    st.info("No categorical columns detected.")

st.markdown("---")

# Missing / duplicates detail
st.subheader("Missing values & duplicates (detail)")

with st.expander("Show rows with missing values"):
    miss_cols = df.columns[df.isnull().any()].tolist()
    if miss_cols:
        st.write("Columns with missing values:", miss_cols)
        st.dataframe(df[df[miss_cols].isnull().any(axis=1)].head(100))
    else:
        st.write("No missing values detected.")

with st.expander("Show duplicate rows (first 50)"):
    if df.duplicated().sum() > 0:
        st.write(f"Duplicate row count: {df.duplicated().sum()}")
        st.dataframe(df[df.duplicated()].head(50))
    else:
        st.write("No duplicate rows found.")

st.markdown("---")

# Quick download cleaned subset (optional)
st.subheader("Export a sample / cleaned subset")
with st.expander("Download sample CSV"):
    take_n = st.number_input("Sample rows to export (0 = full)", min_value=0, max_value=50000, value=1000)
    if st.button("Prepare CSV"):
        if take_n and take_n>0 and take_n < len(df):
            out_df = df.sample(take_n, random_state=42)
        else:
            out_df = df
        csv = out_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="df_sample.csv", mime="text/csv")

st.success("EDA complete — add more plots or feature-specific checks as needed.")
