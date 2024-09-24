import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")


# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text

# Function to process resume text with spaCy
def process_resume_text(resume_text):
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
    return skills

# Function to match resumes with a job description
def match_resumes_with_job(job_description, resumes):
    vectorizer = TfidfVectorizer()
    all_texts = [job_description] + resumes
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity_scores[0]

# Streamlit App
st.title("Resume Matcher")

# Job description input
job_description = st.text_area("Enter Job Description")

# List of PDF paths (Update these paths to your actual PDF locations)
pdf_paths = [
    'C:/Users/proth/Downloads/SAMPLE1.pdf',
    'C:/Users/proth/Downloads/SAMPLE2.pdf',
    'C:/Users/proth/Downloads/SAMPLE3.pdf',
    'C:/Users/proth/Downloads/SAMPLE4.pdf',
    'C:/Users/proth/Downloads/SAMPLE5.pdf'
]

if st.button("Match Resumes"):
    # Extract resume texts
    resume_texts = [extract_text_from_pdf(path) for path in pdf_paths]

    # Match resumes with job description
    similarity_scores = match_resumes_with_job(job_description, resume_texts)

    # Display results
    for i, score in enumerate(similarity_scores):
        st.write(f"Resume {i+1}: Similarity Score = {score:.4f}")
