import streamlit as st
import pandas as pd
import time
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

def ai_match_title(raw_title, standard_titles, threshold=0.8):
    query_embedding = model.encode(raw_title, convert_to_tensor=True)
    standard_embeddings = model.encode(standard_titles, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, standard_embeddings)[0]
    best_match_index = similarities.argmax().item()
    best_score = similarities[best_match_index].item()
    return (standard_titles[best_match_index], best_score) if best_score >= threshold else (None, best_score)

def fuzzy_match_title(raw_title, standard_titles, threshold=80):
    best_match, score, _ = process.extractOne(raw_title, standard_titles, scorer=fuzz.token_sort_ratio)
    return (best_match, score) if best_match else (None, score)

st.title("🔍 Job Title Matcher")
st.subheader("📤 Upload Standardized Job Titles")
uploaded_standard = st.file_uploader("Upload a CSV or Excel file with standardized job titles.", type=["csv", "xlsx"])

if uploaded_standard:
    standard_df = pd.read_csv(uploaded_standard) if uploaded_standard.name.endswith(".csv") else pd.read_excel(uploaded_standard)
    st.session_state.standard_titles = standard_df
    st.success("✅ Standardized job titles uploaded successfully!")

st.subheader("📌 Stored Standardized Job Titles")
st.dataframe(st.session_state.standard_titles, use_container_width=True)

st.subheader("📥 Upload Unclean Job Titles")
uploaded_unclean = st.file_uploader("Upload a CSV or Excel file with unclean job titles.", type=["csv", "xlsx"])

if uploaded_unclean:
    unclean_df = pd.read_csv(uploaded_unclean) if uploaded_unclean.name.endswith(".csv") else pd.read_excel(uploaded_unclean)
    st.session_state.unclean_df = unclean_df
    st.success("✅ Unclean job titles uploaded successfully!")

if st.session_state.standard_titles is not None and st.session_state.unclean_df is not None:
    if st.button("🚀 Run Matching"):
        unclean_titles = st.session_state.unclean_df.iloc[:, 0].astype(str).tolist()
        standard_titles = st.session_state.standard_titles["Standardized Job Title"].astype(str).tolist()

        matched_titles, match_scores = [], []
        progress_bar = st.progress(0)

        for i, title in enumerate(unclean_titles):
            ai_match, ai_score = ai_match_title(title, standard_titles, threshold=0.8)
            if ai_match:
                matched_titles.append(ai_match)
                match_scores.append(ai_score * 100)
            else:
                fuzzy_match, fuzzy_score = fuzzy_match_title(title, standard_titles, threshold=80)
                matched_titles.append(fuzzy_match if fuzzy_match else None)
                match_scores.append(fuzzy_score if fuzzy_match else 0)
            
            progress_bar.progress((i + 1) / len(unclean_titles))
            time.sleep(0.05)  # Simulating progress update
        
        st.session_state.unclean_df["Matched Job Title"] = matched_titles
        st.session_state.unclean_df["Match Score"] = match_scores
        progress_bar.empty()
        st.success("✅ Job titles updated successfully!")

if not st.session_state.mapping_df.empty:
    st.subheader("⚠️ Unmatched Job Titles - Manual Standardization Required")
    for index, row in st.session_state.mapping_df.iterrows():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"❌ **{row['Job Title']}**")
        with col2:
            new_match = st.text_input(f"Enter Standardized Title for: {row['Job Title']}", key=index)
            st.session_state.mapping_df.loc[index, "Matched Job Title"] = new_match if new_match else None

    if st.button("✅ Save Manual Mappings"):
        st.session_state.mapping_df["Match Score"] = 100
        st.session_state.unclean_df.update(st.session_state.mapping_df)
        st.success("✔ Manual mappings saved successfully!")

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
