import streamlit as st
import pandas as pd
import spacy
import base64
import sqlite3
import re
import io
import nltk
import yt_dlp
from PIL import Image
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from streamlit_tags import st_tags
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import plotly.express as px

# --- Pre-run Setup: NLTK Downloader and SpaCy Model Loading ---

# Function to download NLTK data if not present
def download_nltk_data():
    """Downloads required NLTK data packages."""
    packages = ['stopwords', 'punkt']
    for package in packages:
        try:
            nltk.data.find(f'corpora/{package}' if package == 'stopwords' else f'tokenizers/{package}')
        except nltk.downloader.DownloadError:
            nltk.download(package)

# Download data and load the SpaCy model at the start
download_nltk_data()
nlp = spacy.load('en_core_web_sm')

# --- Helper Functions ---

def pdf_reader(file):
    """Extracts text from a PDF file."""
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    """Displays a PDF file in the app."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def get_youtube_video_info(video_url):
    """Fetches YouTube video title and thumbnail using yt-dlp."""
    ydl_opts = {'quiet': True, 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
            title = info.get('title', 'N/A')
            thumbnail = info.get('thumbnail', None)
            return title, thumbnail
        except yt_dlp.utils.DownloadError:
            return "Video not available", None

# --- Database Functions ---

def create_connection():
    """Creates a connection to the SQLite database."""
    return sqlite3.connect('resume_data.db')

def create_table():
    """Creates the user_data table if it doesn't exist."""
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            resume_score REAL,
            timestamp DATETIME,
            page_no INTEGER,
            predicted_field TEXT,
            user_level TEXT,
            actual_skills TEXT,
            recommended_skills TEXT,
            recommended_courses TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_data(name, email, resume_score, predicted_field, user_level, actual_skills, recommended_skills, recommended_courses):
    """Inserts analysis data into the database."""
    conn = create_connection()
    c = conn.cursor()
    timestamp = pd.to_datetime('now')
    c.execute(
        "INSERT INTO user_data (name, email, resume_score, timestamp, page_no, predicted_field, user_level, actual_skills, recommended_skills, recommended_courses) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (name, email, resume_score, timestamp, 1, predicted_field, user_level, str(actual_skills), str(recommended_skills), str(recommended_courses))
    )
    conn.commit()
    conn.close()

# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Resume Analyzer", layout="wide")
    
    # Create database table on first run
    create_table()

    st.title("AI-Powered Resume Analyzer 💡")
    st.sidebar.markdown("# Choose a Page")
    page = st.sidebar.radio("Navigation", ["Resume Analysis", "Admin"])

    if page == "Resume Analysis":
        # Placeholder for resume analysis logic (pyresparser, etc.)
        # This part requires the pyresparser library to be fully implemented.
        # For now, we'll focus on the structure.
        st.header("Upload Your Resume")
        uploaded_file = st.file_uploader("Upload your resume in PDF format", type=['pdf'])

        if uploaded_file:
            st.success("File uploaded successfully!")
            # This is where you would call pyresparser to get skills, name, etc.
            # Example placeholder data:
            resume_data = {
                'name': 'John Doe',
                'email': 'john.doe@email.com',
                'skills': ['Python', 'Data Analysis', 'Machine Learning', 'SQL']
            }
            st.write(f"**Name:** {resume_data['name']}")
            st.write(f"**Email:** {resume_data['email']}")
            st.write(f"**Skills:** {', '.join(resume_data['skills'])}")

            # Placeholder for analysis and recommendations
            st.header("Analysis & Recommendations")
            st.write("Based on your resume, we recommend the following...")
            
            # --- Recommendations Section ---
            st.subheader("Recommended Courses")
            for course_name, course_link in ds_course.items():
                st.markdown(f"[{course_name}]({course_link})")
            
            st.subheader("Resume Improvement Videos")
            for video_url in resume_videos:
                title, thumbnail = get_youtube_video_info(video_url)
                if thumbnail:
                    st.image(thumbnail, width=300)
                    st.markdown(f"[{title}]({video_url})")
                else:
                    st.warning(f"Could not load video: {title}")
    
    elif page == "Admin":
        st.header("Admin Panel")
        conn = create_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM user_data", conn)
            st.dataframe(df)
            
            # --- Admin Dashboard Visuals ---
            if not df.empty:
                st.subheader("Analytics Dashboard")
                
                # Predicted Field Distribution
                field_counts = df['predicted_field'].value_counts()
                fig1 = px.pie(field_counts, values=field_counts.values, names=field_counts.index, title="Distribution of Candidate Fields")
                st.plotly_chart(fig1)

                # Resume Score Distribution
                fig2 = px.histogram(df, x='resume_score', title='Distribution of Resume Scores')
                st.plotly_chart(fig2)
        
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
        finally:
            conn.close()

if __name__ == '__main__':
    main()
