from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
JD_FOLDER = os.path.join(UPLOAD_FOLDER, 'job_description')
CANDIDATE_FOLDER = os.path.join(UPLOAD_FOLDER, 'candidate_resumes')

# Create the folders if they don't exist
os.makedirs(JD_FOLDER, exist_ok=True)
os.makedirs(CANDIDATE_FOLDER, exist_ok=True)

# API routes
@app.route('/')
def hello_world(): 
    return 'Hello, World!'

#! POST Route for Job Description
@app.route('/upload_job_description', methods=['POST'])
def upload_job_description():
    file = request.files.get('pdf') # same key used in React: 'pdf'

    if not file:
        return jsonify({"error":"Job description not uploaded"}), 400

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(JD_FOLDER, file.filename)
        file.save(filepath)
        return jsonify({"message": "Job description uploaded", "filename": file.filename}), 200

    return jsonify({"error": "Invalid file type"}), 400


#! POST Route for Cadidate Resumes
@app.route('/upload_candidate_resumes',methods=["POST"])
def upload_multiple_pdfs():
    files = request.files.getlist('pdfs')  # same key used in React: 'pdfs'

    if not files or len(files) == 0:
        return jsonify({"error": "No files uploaded"}), 400

    saved_files = []

    for file in files:
        if file.filename.endswith('.pdf'):
            filepath = os.path.join(CANDIDATE_FOLDER, file.filename)
            file.save(filepath)
            saved_files.append(file.filename)

    return jsonify({
        "message": f"{len(saved_files)} PDFs uploaded successfully!",
        "files": saved_files
    }), 200


#? Processing the Text
# Download stopwords if not already done
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def clean_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove special characters (keep only letters and numbers, spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 3. Tokenize by splitting on whitespace
    words = text.split()

    # 4. Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join back to string
    return " ".join(words)



#! GET Route for final rankings
@app.route('/calculate_ranks',methods=["GET"])
def calculate_resume_rank():
    #* Loading the Candidate Resumes
    folder_path = CANDIDATE_FOLDER
    all_resume_texts = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):  # Only process PDF files
            file_path = os.path.join(folder_path, filename)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            all_resume_texts.append((filename, text))

    #* Loading the Job Description
    job_path = os.path.join(JD_FOLDER,'job_description.pdf')
    job_doc = fitz.open(job_path)
    job_text = ""
    for page in job_doc:
        job_text += page.get_text()

    # Clean all resumes
    cleaned_resumes = [(fname, clean_text(txt)) for fname, txt in all_resume_texts]
    # Clean job description
    cleaned_job_desc = clean_text(job_text)

    #* Vetorization
    # Combine all cleaned resumes and job description texts into a list
    documents = [cleaned_job_desc] + [txt for fname, txt in cleaned_resumes]
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the documents to get TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Let’s say first document is the job description:
    job_desc_vector = tfidf_matrix[1]  # first row
    resume_vectors = tfidf_matrix[1:]  # all except first row

    # Calculate cosine similarity between job description and each resume
    similarity_scores = cosine_similarity(resume_vectors, job_desc_vector)

    # Flatten similarity array
    scores = similarity_scores.flatten()

    # Combine filenames with scores
    results = [
        {"filename": fname, "score": score}
        for (fname, _), score in zip(cleaned_resumes, scores)
    ]

    return jsonify(results), 200




if __name__ == '__main__':
    app.run(debug=True)