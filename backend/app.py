import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Automated Resume Screening System", layout="wide")

st.title("Automated Resume Screening System")
st.write("Analyze resumes, score candidates, match jobs, check ATS compatibility, and rank applicants.")

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def extract_text_from_docx(file):
    document = docx.Document(file)
    return " ".join([para.text for para in document.paragraphs])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r"(\+?\d[\d\s\-\(\)]{8,}\d)", text)
    return match.group(0) if match else "Not found"

def ats_score(text):
    score = 0
    checks = {
        "Email present": extract_email(text) != "Not found",
        "Phone present": extract_phone(text) != "Not found",
        "Skills section": "skills" in text.lower(),
        "Experience section": "experience" in text.lower(),
        "Education section": "education" in text.lower(),
        "Projects section": "project" in text.lower(),
        "Simple formatting": len(re.findall(r"[^\x00-\x7F]", text)) < 20
    }

    for passed in checks.values():
        if passed:
            score += 100 / len(checks)

    return round(score, 2), checks

def keyword_score(resume_text, job_text):
    job_words = set(clean_text(job_text).split())
    resume_words = set(clean_text(resume_text).split())

    if len(job_words) == 0:
        return 0

    matched = job_words.intersection(resume_words)
    return round((len(matched) / len(job_words)) * 100, 2), matched

def similarity_score(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

job_description = st.text_area("Paste Job Description Here", height=250)

uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if st.button("Analyze Resumes"):
    if not job_description:
        st.error("Please paste a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        results = []

        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(file)
            elif file.name.endswith(".docx"):
                resume_text = extract_text_from_docx(file)
            else:
                resume_text = file.read().decode("utf-8", errors="ignore")

            cleaned_resume = clean_text(resume_text)
            cleaned_job = clean_text(job_description)

            sim_score = similarity_score(cleaned_resume, cleaned_job)
            key_score, matched_keywords = keyword_score(resume_text, job_description)
            ats, ats_checks = ats_score(resume_text)

            final_score = round((sim_score * 0.5) + (key_score * 0.3) + (ats * 0.2), 2)

            results.append({
                "Candidate File": file.name,
                "Email": extract_email(resume_text),
                "Phone": extract_phone(resume_text),
                "Job Match Score": sim_score,
                "Keyword Score": key_score,
                "ATS Score": ats,
                "Final Score": final_score,
                "Matched Keywords": ", ".join(list(matched_keywords)[:20])
            })

        df = pd.DataFrame(results)
        df = df.sort_values(by="Final Score", ascending=False)

        st.subheader("Ranked Applicants")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results as CSV",
            csv,
            "resume_screening_results.csv",
            "text/csv"
        )

        st.subheader("Top Candidate")
        top = df.iloc[0]
        st.success(f"Best Match: {top['Candidate File']} with Final Score {top['Final Score']}%")