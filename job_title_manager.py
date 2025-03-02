import streamlit as st
import pandas as pd
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# Set Streamlit theme for a modern high-tech look
st.set_page_config(page_title="Job Title Matcher", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
        }
        .block-container {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "standard_titles" not in st.session_state:
    st.session_state.standard_titles = pd.DataFrame(columns=["Standardized Job Title"])
if "unclean_df" not in st.session_state:
    st.session_state.unclean_df = None
if "mapping_df" not in st.session_state:
    st.session_state.mapping_df = pd.DataFrame(columns=["Job Title", "Matched Job Title"])

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

def match_job_titles(unclean_titles, standard_titles, progress_bar):
    """Matches job titles using AI embedding and fuzzy matching."""
    unclean_embeddings = model.encode(unclean_titles, convert_to_tensor=True)
    standard_embeddings = model.encode(standard_titles, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(unclean_embeddings, standard_embeddings)
    best_match_indices = similarities.argmax(dim=1)
    best_scores = similarities.max(dim=1).values.cpu().numpy()
    
    standard_title_lookup = {i: standard_titles[i] for i in range(len(standard_titles))}
    matched_titles = [standard_title_lookup[idx] if score >= 0.75 else None for idx, score in zip(best_match_indices, best_scores)]
    
    for i in range(len(matched_titles)):
        if matched_titles[i] is None:
            fuzzy_result = process.extractOne(unclean_titles[i], standard_titles, scorer=fuzz.WRatio)
            if fuzzy_result:
                fuzzy_match, fuzzy_score, _ = fuzzy_result  # Correct unpacking
                if fuzzy_score >= 80:
                    matched_titles[i] = fuzzy_match
                    best_scores[i] = fuzzy_score / 100
    
    progress_bar.progress(100)
    return matched_titles, best_scores * 100

st.title("🔍 Job Title Matcher")
st.subheader("📤 Upload Standardized Job Titles")
uploaded_standard = st.file_uploader("Upload a CSV or Excel file with standardized job titles.", type=["csv", "xlsx"])

if uploaded_standard:
    standard_df = pd.read_csv(uploaded_standard) if uploaded_standard.name.endswith(".csv") else pd.read_excel(uploaded_standard)
    st.session_state.standard_titles = standard_df
    st.success("✅ Standardized job titles uploaded successfully!")

st.subheader("📥 Upload Unclean Job Titles")
uploaded_unclean = st.file_uploader("Upload a CSV or Excel file with unclean job titles.", type=["csv", "xlsx"])

if uploaded_unclean:
    unclean_df = pd.read_csv(uploaded_unclean) if uploaded_unclean.name.endswith(".csv") else pd.read_excel(uploaded_unclean)
    st.session_state.unclean_df = unclean_df
    st.success("✅ Unclean job titles uploaded successfully!")

if st.session_state.standard_titles is not None and st.session_state.unclean_df is not None:
    if st.button("🚀 Run Matching"):
        progress_bar = st.progress(0)
        unclean_titles = st.session_state.unclean_df.iloc[:, 0].astype(str).tolist()
        standard_titles = st.session_state.standard_titles["Standardized Job Title"].astype(str).tolist()
        
        matched_titles, match_scores = match_job_titles(unclean_titles, standard_titles, progress_bar)
        st.session_state.unclean_df["Matched Job Title"] = matched_titles
        st.session_state.unclean_df["Match Score"] = match_scores
        st.session_state.mapping_df = st.session_state.unclean_df.copy()
        st.success("✅ Job titles updated successfully! Please review before exporting.")

if not st.session_state.mapping_df.empty:
    st.subheader("🛠 Review & Edit Mapped Job Titles")
    for index, row in st.session_state.mapping_df.iterrows():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"📝 **{row['Job Title']}**")
        with col2:
            new_match = st.text_input(f"Edit Matched Title for: {row['Job Title']}", value=row['Matched Job Title'], key=index)
            st.session_state.mapping_df.at[index, "Matched Job Title"] = new_match
    if st.button("✅ Save Changes"):
        st.session_state.unclean_df.update(st.session_state.mapping_df)
        st.success("✔ Changes saved successfully!")

st.subheader("📥 Finalize Mapping & Export")
if st.button("📥 Export Updated File"):
    final_cleaned_file = "final_cleaned_job_titles.csv"
    st.session_state.unclean_df.to_csv(final_cleaned_file, index=False)
    st.download_button(
        label="📥 Download Updated Job Titles",
        data=st.session_state.unclean_df.to_csv(index=False),
        file_name="final_cleaned_job_titles.csv",
        mime="text/csv"
    )
    st.success("🎉 Mapping finalized and ready for export!")
