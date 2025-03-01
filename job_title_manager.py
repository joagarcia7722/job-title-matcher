import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# Initialize session state for standardized job titles and mappings
if "standard_titles" not in st.session_state:
    st.session_state.standard_titles = pd.DataFrame(columns=["Standardized Job Title"])
if "unclean_df" not in st.session_state:
    st.session_state.unclean_df = None
if "mapping_df" not in st.session_state:
    st.session_state.mapping_df = pd.DataFrame(columns=["Job Title", "Matched Job Title"])

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to match job titles using AI embeddings
def ai_match_title(raw_title, standard_titles, threshold=0.8):
    query_embedding = model.encode(raw_title, convert_to_tensor=True)
    standard_embeddings = model.encode(standard_titles, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, standard_embeddings)[0]
    best_match_index = similarities.argmax().item()
    best_score = similarities[best_match_index].item()

    return (standard_titles[best_match_index], best_score) if best_score >= threshold else (None, best_score)

# Function to match job titles using fuzzy matching
def fuzzy_match_title(raw_title, standard_titles, threshold=80):
    best_match, score, _ = process.extractOne(raw_title, standard_titles, scorer=fuzz.token_sort_ratio)
    return (best_match, score) if best_match else (None, score)

# UI Header
st.title("📝 Job Title Standardization Tool")

# **Upload Standardized Job Titles**
st.subheader("📤 Upload Standardized Job Titles")
uploaded_standard = st.file_uploader("Upload a CSV or Excel file with standardized job titles.", type=["csv", "xlsx"])

if uploaded_standard:
    if uploaded_standard.name.endswith(".csv"):
        standard_df = pd.read_csv(uploaded_standard)
    else:
        standard_df = pd.read_excel(uploaded_standard)

    st.session_state.standard_titles = standard_df
    st.success("✅ Standardized job titles uploaded successfully!")

# Show stored standardized job titles
st.subheader("📌 Stored Standardized Job Titles")
st.dataframe(st.session_state.standard_titles)

# **Upload Unclean Job Titles**
st.subheader("📥 Upload Unclean Job Titles")
uploaded_unclean = st.file_uploader("Upload a CSV or Excel file with unclean job titles.", type=["csv", "xlsx"])

if uploaded_unclean:
    if uploaded_unclean.name.endswith(".csv"):
        unclean_df = pd.read_csv(uploaded_unclean)
    else:
        unclean_df = pd.read_excel(uploaded_unclean)

    st.session_state.unclean_df = unclean_df
    st.success("✅ Unclean job titles uploaded successfully!")

# **Run Button to Process Matching**
if st.session_state.standard_titles is not None and st.session_state.unclean_df is not None:
    if st.button("🚀 Run Matching"):
        unclean_titles = st.session_state.unclean_df.iloc[:, 0].astype(str).tolist()  # Assume job titles are in the first column
        standard_titles = st.session_state.standard_titles["Standardized Job Title"].astype(str).tolist()

        matched_titles = []
        match_scores = []

        for title in unclean_titles:
            ai_match, ai_score = ai_match_title(title, standard_titles, threshold=0.8)

            if ai_match:
                matched_titles.append(ai_match)
                match_scores.append(ai_score * 100)  # Convert to percentage
            else:
                fuzzy_match, fuzzy_score = fuzzy_match_title(title, standard_titles, threshold=80)
                if fuzzy_match:
                    matched_titles.append(fuzzy_match)
                    match_scores.append(fuzzy_score)
                else:
                    matched_titles.append(None)
                    match_scores.append(0)

        # Store results
        st.session_state.unclean_df["Matched Job Title"] = matched_titles
        st.session_state.unclean_df["Match Score"] = match_scores

        # Separate unmatched job titles
        unmatched_df = st.session_state.unclean_df[st.session_state.unclean_df["Matched Job Title"].isnull()]
        st.session_state.mapping_df = unmatched_df.copy()

        st.success("✅ Job titles updated successfully!")

# **Show Unmatched Job Titles for Manual Mapping**
if not st.session_state.mapping_df.empty:
    st.subheader("⚠️ Unmatched Job Titles - Manual Standardization Required")
    updated_titles = []

    for index, row in st.session_state.mapping_df.iterrows():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"❌ **{row['Job Title']}**")
        with col2:
            new_match = st.text_input(f"Enter Standardized Title for: {row['Job Title']}", key=index)
            updated_titles.append(new_match if new_match else None)

    # Save manual updates
    if st.button("✅ Save Manual Mappings"):
        st.session_state.mapping_df["Matched Job Title"] = updated_titles
        st.session_state.mapping_df["Match Score"] = 100
        st.session_state.unclean_df.update(st.session_state.mapping_df)
        st.success("✔ Manual mappings saved successfully!")

# **Update Existing Mappings**
st.subheader("🔄 Update Existing Mappings")
job_to_update = st.text_input("Enter job title to update its mapping:")
if job_to_update:
    new_mapping = st.text_input(f"Enter new standard title for '{job_to_update}':")
    if st.button("🔄 Update Mapping"):
        st.session_state.unclean_df.loc[st.session_state.unclean_df["Job Title"] == job_to_update, "Matched Job Title"] = new_mapping
        st.session_state.unclean_df.loc[st.session_state.unclean_df["Job Title"] == job_to_update, "Match Score"] = 100
        st.success(f"✔ Updated mapping for '{job_to_update}'!")

# **Finalize and Export Mappings**
if st.button("📥 Finalize Mapping & Export Updated File"):
    final_cleaned_file = "cleaned_job_titles.csv"
    st.session_state.unclean_df.to_csv(final_cleaned_file, index=False)

    st.download_button(
        label="📥 Download Updated Job Titles",
        data=st.session_state.unclean_df.to_csv(index=False),
        file_name="final_cleaned_job_titles.csv",
        mime="text/csv"
    )
    st.success("🎉 Mapping finalized and ready for export!")

