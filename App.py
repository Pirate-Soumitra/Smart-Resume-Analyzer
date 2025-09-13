import streamlit as st
import spacy
import pandas as pd
import base64
import time
import datetime
import io
import os
import random
import sqlite3
import re  # Import the regular expression module
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import plotly.express as px
import yt_dlp

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon='./Logo/SRA_Logo.ico',
)

# --- Constants ---
SKILLS_DB = {
    'Data Science': ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit'],
    'Web Development': ['react', 'django', 'node js', 'react js', 'php', 'laravel', 'magento', 'wordpress', 'javascript', 'angular js', 'c#', 'flask'],
    'Android Development': ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy'],
    'IOS Development': ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode'],
    'UI-UX Development': ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes', 'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator', 'illustrator', 'adobe after effects', 'after effects', 'adobe premier pro', 'premier pro', 'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp', 'user research', 'user experience']
}
RECOMMENDED_SKILLS = {
    'Data Science': ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras', 'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask", 'Streamlit'],
    'Web Development': ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento', 'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK'],
    'Android Development': ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java', 'Kivy', 'GIT', 'SDK', 'SQLite'],
    'IOS Development': ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode', 'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation', 'Auto-Layout'],
    'UI-UX Development': ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq', 'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing', 'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe', 'Solid', 'Grasp', 'User Research']
}
COURSE_RECOMMENDATION = {
    'Data Science': ds_course,
    'Web Development': web_course,
    'Android Development': android_course,
    'IOS Development': ios_course,
    'UI-UX Development': uiux_course
}

# --- Caching ---
@st.cache_resource
def get_db_connection():
    """Initializes the SQLite database and returns the connection."""
    conn = sqlite3.connect('sra.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_data
                 (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                  Name TEXT NOT NULL,
                  Email_ID TEXT NOT NULL,
                  resume_score TEXT NOT NULL,
                  Timestamp TEXT NOT NULL,
                  Page_no TEXT NOT NULL,
                  Predicted_Field TEXT NOT NULL,
                  User_level TEXT NOT NULL,
                  Actual_skills TEXT NOT NULL,
                  Recommended_skills TEXT NOT NULL,
                  Recommended_courses TEXT NOT NULL);''')
    conn.commit()
    return conn

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model 'en_core_web_sm' and caches it."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        st.error("Missing spaCy model. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
        st.stop()

# --- Database Operations ---
def insert_data(connection, name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    """Inserts the resume analysis data into the SQLite database."""
    cursor = connection.cursor()
    insert_sql = """INSERT INTO user_data (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    rec_values = (name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, str(skills), str(recommended_skills), str(courses))
    cursor.execute(insert_sql, rec_values)
    connection.commit()

# --- Helper Functions ---
def fetch_yt_video(link):
    """Fetches YouTube video title using yt-dlp."""
    try:
        ydl_opts = {'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(link, download=False)
            return info_dict.get('title', 'YouTube Video')
    except Exception:
        return "YouTube Video"

def get_table_download_link(df, filename, text):
    """Generates a link to download a pandas DataFrame as a CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def pdf_reader(file):
    """Reads text from a PDF file path."""
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
    """Displays a PDF file in the Streamlit app."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    """Recommends courses to the user."""
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for i, (c_name, c_link) in enumerate(course_list[:no_of_reco], 1):
        st.markdown(f"({i}) [{c_name}]({c_link})")
        rec_course.append(c_name)
    return rec_course

# --- NEW: Custom Resume Parser ---
def extract_data_from_resume(file_path, nlp_model):
    """Extracts information from a resume using spaCy and regex."""
    # 1. Extract text and page count from PDF
    resume_text = pdf_reader(file_path)
    page_count = 0
    with open(file_path, 'rb') as fh:
        page_count = len(list(PDFPage.get_pages(fh, check_extractable=False)))

    # 2. Process text with spaCy
    doc = nlp_model(resume_text)

    # 3. Extract Name, Email, and Mobile Number
    name, email, mobile_number = '', '', ''
    
    # Extract Name (usually the first PERSON entity)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            name = ent.text
            break
            
    # Extract Email using regex
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', resume_text)
    if email_match:
        email = email_match.group(0)

    # Extract Mobile Number using a more flexible regex
    phone_match = re.search(r'(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', resume_text)
    if phone_match:
        mobile_number = phone_match.group(0)

    # 4. Extract Skills
    all_skills_set = set([skill.lower() for sublist in SKILLS_DB.values() for skill in sublist])
    found_skills = []
    for skill in all_skills_set:
        if re.search(r'\b' + re.escape(skill) + r'\b', resume_text, re.IGNORECASE):
            found_skills.append(skill.capitalize())

    # 5. Return data in a dictionary mimicking pyresparser's output
    return {
        'name': name,
        'email': email,
        'mobile_number': mobile_number,
        'skills': list(set(found_skills)),  # Use set to get unique skills
        'no_of_pages': page_count,
    }

def analyze_resume(file_path, nlp_model):
    """Parses resume, analyzes content, and returns a dictionary of results."""
    # --- MODIFIED: Call our custom function instead of ResumeParser ---
    resume_data = extract_data_from_resume(file_path, nlp_model)
    
    if not resume_data:
        return None

    # Get resume text for analysis (we can reuse the function)
    resume_text = pdf_reader(file_path)

    # Determine candidate level
    if resume_data.get('no_of_pages', 0) <= 1:
        cand_level = "Fresher"
    elif resume_data.get('no_of_pages', 0) == 2:
        cand_level = "Intermediate"
    else:
        cand_level = "Experienced"

    # Field and skills recommendation
    reco_field = ''
    recommended_skills = []
    rec_course_list = []
    
    resume_skills = resume_data.get('skills', []) or []
    for field, keywords in SKILLS_DB.items():
        if any(skill.lower() in keywords for skill in resume_skills):
            reco_field = field
            recommended_skills = RECOMMENDED_SKILLS.get(field, [])
            rec_course_list = COURSE_RECOMMENDATION.get(field, [])
            break
    
    # Calculate resume score
    resume_score = 0
    score_criteria = {
        'Objective': 20,
        'Declaration': 20,
        'Hobbies': 20,
        'Interests': 0,
        'Achievements': 20,
        'Projects': 20
    }
    found_criteria = set()
    for criterion, score in score_criteria.items():
        if re.search(r'\b' + criterion + r'\b', resume_text, re.IGNORECASE):
            if criterion == 'Interests' and 'Hobbies' in found_criteria:
                continue
            resume_score += score
            found_criteria.add(criterion)

    return {
        "resume_data": resume_data,
        "resume_text": resume_text,
        "cand_level": cand_level,
        "reco_field": reco_field,
        "recommended_skills": recommended_skills,
        "rec_course_list": rec_course_list,
        "resume_score": resume_score
    }

# --- UI Functions ---
def handle_normal_user(connection, nlp_model):
    """Handles the UI and logic for the 'Normal User' role."""
    st.subheader("Upload your Resume to get started")
    pdf_file = st.file_uploader("Choose your Resume (PDF only)", type=["pdf"])

    if pdf_file is not None:
        save_path = './Uploaded_Resumes/'
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, pdf_file.name)
        with open(save_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        show_pdf(save_file_path)
        
        analysis_results = analyze_resume(save_file_path, nlp_model)
        
        if not analysis_results or not analysis_results['resume_data']:
            st.error("Sorry, something went wrong while parsing the resume.")
            return

        resume_data = analysis_results['resume_data']
        
        st.header("**Resume Analysis**")
        st.success(f"Hello {resume_data.get('name', 'User')}")
        st.subheader("**Your Basic Info**")
        try:
            st.text(f"Name: {resume_data['name']}")
            st.text(f"Email: {resume_data['email']}")
            st.text(f"Contact: {resume_data['mobile_number']}")
            st.text(f"Resume pages: {resume_data['no_of_pages']}")
        except KeyError as e:
            st.warning(f"Could not find: {e.args[0]}")
        
        cand_level = analysis_results['cand_level']
        level_messages = {
            "Fresher": "<h4 style='text-align: left; color: #d73b5c;'>You seem to be a Fresher.</h4>",
            "Intermediate": "<h4 style='text-align: left; color: #1ed760;'>You are at an intermediate level!</h4>",
            "Experienced": "<h4 style='text-align: left; color: #fba171;'>You have an experienced profile!</h4>"
        }
        st.markdown(level_messages.get(cand_level, ''), unsafe_allow_html=True)
        
        st.subheader("**Skills Recommendation💡**")
        st_tags(label='### Your Skills', text='Skills from your resume', value=resume_data.get('skills', []), key='user_skills')
        
        reco_field = analysis_results['reco_field']
        if reco_field:
            st.success(f"** Our analysis suggests you are looking for {reco_field} Jobs.**")
            st_tags(label='### Recommended Skills', text='Add these to your resume', value=analysis_results['recommended_skills'], key='recommended_skills')
            st.markdown("<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 your chances of getting a Job💼</h4>", unsafe_allow_html=True)
            
            rec_course_names = course_recommender(analysis_results['rec_course_list'])
        else:
            st.warning("Could not determine a specific job field from your skills.")
            rec_course_names = []

        st.subheader("**Resume Score & Tips💡**")
        resume_score = analysis_results['resume_score']
        tips = {
            'Objective': ('[+] Awesome! You have added an Objective', '[-] Please add a career objective to state your intentions to recruiters.'),
            'Declaration': ('[+] Awesome! You have added a Declaration✍️', '[-] Please add a Declaration to assure that everything written is true.'),
            'Hobbies': ('[+] Awesome! You have added your Hobbies⚽', '[-] Please add Hobbies/Interests to show your personality.'),
            'Achievements': ('[+] Awesome! You have added your Achievements🏅', '[-] Please add Achievements to showcase your capabilities.'),
            'Projects': ('[+] Awesome! You have added your Projects👨‍💻', '[-] Please add Projects to demonstrate relevant work experience.')
        }
        
        for tip_name, messages in tips.items():
            if re.search(r'\b' + tip_name + r'\b', analysis_results['resume_text'], re.IGNORECASE):
                st.markdown(f"<h4 style='text-align: left; color: #1ed760;'>{messages[0]}</h4>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h4 style='text-align: left; color: #fabc10;'>{messages[1]}</h4>", unsafe_allow_html=True)

        st.subheader("**Resume Score📝**")
        st.markdown("""<style>.stProgress > div > div > div > div {background-color: #d73b5c;}</style>""", unsafe_allow_html=True)
        my_bar = st.progress(0)
        score_to_display = min(100, resume_score)
        for percent_complete in range(score_to_display + 1):
            time.sleep(0.01)
            my_bar.progress(percent_complete)
        st.success(f'** Your Resume Score: {score_to_display} / 100**')
        st.warning("** Note: This score is calculated based on the presence of key sections in your resume. **")
        st.balloons()
        
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        insert_data(connection, resume_data.get('name'), resume_data.get('email'), score_to_display, timestamp,
                    resume_data.get('no_of_pages'), reco_field, cand_level, resume_data.get('skills'),
                    analysis_results['recommended_skills'], rec_course_names)
        
        st.header("**Bonus Videos for You💡**")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Resume Writing Tips")
            resume_vid = random.choice(resume_videos)
            res_vid_title = fetch_yt_video(resume_vid)
            st.video(resume_vid)
            st.write(f"✅ **{res_vid_title}**")
        with col2:
            st.subheader("Interview Tips")
            interview_vid = random.choice(interview_videos)
            int_vid_title = fetch_yt_video(interview_vid)
            st.video(interview_vid)
            st.write(f"✅ **{int_vid_title}**")

def handle_admin(connection):
    """Handles the UI and logic for the 'Admin' role."""
    st.success('Welcome to the Admin Dashboard')
    ad_user = st.text_input("Username")
    ad_password = st.text_input("Password", type='password')
    if st.button('Login'):
        if ad_user == 'admin' and ad_password == 'admin123':
            st.success("Welcome Admin!")
            
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM user_data")
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Pages',
                                             'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                             'Recommended Courses'])
            st.header("**User's Data**")
            st.dataframe(df)
            st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report as CSV'), unsafe_allow_html=True)
            
            plot_data = pd.read_sql("SELECT Predicted_Field, User_level FROM user_data;", connection)
            
            st.header("📈 Data Visualization")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("**Predicted Fields**")
                fig = px.pie(plot_data, names='Predicted_Field', title='Distribution of Predicted Fields')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.subheader("**User Experience Levels**")
                fig = px.pie(plot_data, names='User_level', title="Distribution of User Experience Levels")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Wrong Username or Password")

def run():
    """Main function to run the Streamlit app."""
    connection = get_db_connection()
    nlp_model = load_spacy_model()

    st.title("Smart Resume Analyser")
    st.sidebar.markdown("# Choose User Role")
    activities = ["Normal User", "Admin"]
    choice = st.sidebar.selectbox("Select your role:", activities)
    
    try:
        img = Image.open('./Logo/SRA_Logo.jpg')
        st.image(img.resize((250, 250)))
    except FileNotFoundError:
        st.warning("Logo file not found. Please make sure './Logo/SRA_Logo.jpg' exists.")

    if choice == 'Normal User':
        handle_normal_user(connection, nlp_model)
    elif choice == 'Admin':
        handle_admin(connection)

if __name__ == '__main__':
    run()