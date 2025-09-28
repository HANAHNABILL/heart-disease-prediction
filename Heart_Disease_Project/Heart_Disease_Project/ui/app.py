# Heart Disease Prediction App
import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CardioPredict Pro - Heart Disease Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#===================CSS=====================
st.markdown("""
<style>
    /* Main app styling */
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    
    /* Card containers */
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #222222 !important; /* Force text color in light mode */
    }
    
    .card h1, .card h2, .card h3, .card h4, .card h5, .card h6 {
        color: #2E86AB !important;
    }
    
    .card p, .card li, .card span {
        color: #444444 !important;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .viz-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        color: #222222 !important;
    }
    
    /* Clinical notes with improved styling */
    .clinical-note {
        background: linear-gradient(135deg, #FFF9E6 0%, #FFF3CD 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #F18F01;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        color: #222222 !important;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #FFE6E6 0%, #FFCCCC 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #D00000;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        color: #222222 !important;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #E6FFE6 0%, #CCFFCC 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #38B000;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        color: #222222 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }
    
    /* Form elements */
    .stSlider, .stSelectbox, .stRadio {
        margin: 0.8rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white !important;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-card h3, .metric-card h2, .metric-card p {
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
    }
    
    /* Custom expander */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2E86AB;
    }
    
    /* Custom navigation buttons */
    .nav-button {
        display: block;
        width: 100%;
        padding: 12px 16px;
        margin: 8px 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: none;
        border-radius: 8px;
        text-align: left;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        color: #2E86AB !important;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white !important;
        transform: translateX(5px);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }
    
    /* Ensure all text in main content is visible */
    .main .block-container {
        color: #222222 !important;
    }
    
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #2E86AB !important;
    }
    
    .main p, .main li, .main span, .main div {
        color: #444444 !important;
    }
</style>
""", unsafe_allow_html=True)

#===================Clinical override function==================================================
def clinical_override(age, thalach, raw_probability, sex=None, cp=None, exang=None,
                     oldpeak=None, thal=None, fbs=None, trestbps=None, chol=None):
    """
    Apply clinical sanity checks to model predictions with enhanced high-risk pattern detection
    """
    predicted_max = 220 - age
    hr_percentage = thalach / predicted_max

    clinical_notes = []
    adjusted_prob = raw_probability
    # Young patient with abnormally low HR
    if age < 40 and hr_percentage < 0.65:
        max_reasonable_risk = 0.3
        adjusted_prob = min(adjusted_prob, max_reasonable_risk)
        clinical_notes.append(f"⚠️ Unusually low exercise capacity for age (achieved {thalach}bpm vs expected {predicted_max}bpm). Risk capped due to clinical rarity.")

    # Elderly patient with unusually high HR
    if age > 60 and hr_percentage > 1.1:
        max_reasonable_risk = 0.4
        adjusted_prob = min(adjusted_prob, max_reasonable_risk)
        clinical_notes.append(f"⚠️ Unusually high exercise capacity for age. Risk adjusted due to clinical rarity.")

    high_risk_patterns = 0
    pattern_details = []

    # Pattern 1: Typical angina + exercise angina + significant ST depression
    if cp and exang and oldpeak is not None:
        if cp == "Typical Angina" and exang == "Yes" and oldpeak >= 2.0:
            high_risk_patterns += 1
            adjustment_factor = 1.3 if oldpeak >= 3.0 else 1.2
            adjusted_prob = min(adjusted_prob * adjustment_factor, 0.95)
            pattern_details.append(f"Typical angina + exercise-induced angina + ST depression ({oldpeak}mm)")

    # Pattern 2: Multiple metabolic risk factors
    metabolic_risks = 0
    if trestbps and trestbps >= 140:
        metabolic_risks += 1
    if chol and chol >= 240:
        metabolic_risks += 1
    if fbs and fbs == "Yes":
        metabolic_risks += 1

    if metabolic_risks >= 2:
        high_risk_patterns += 1
        adjusted_prob = min(adjusted_prob * (1 + metabolic_risks * 0.15), 0.90)
        pattern_details.append(f"Multiple metabolic risk factors ({metabolic_risks}/3)")

    # Pattern 3: High-risk thalassemia findings
    if thal and thal in ["Fixed Defect", "Reversible Defect"]:
        high_risk_patterns += 1
        if thal == "Reversible Defect" and oldpeak and oldpeak > 2.0:
            adjusted_prob = min(adjusted_prob * 1.25, 0.95)
            pattern_details.append("Reversible defect with significant ST changes")
        else:
            adjusted_prob = min(adjusted_prob * 1.15, 0.90)
            pattern_details.append(f"{thal} identified")

    # Pattern 4: Elderly diabetic female (high-risk demographic)
    if sex and age and fbs:
        if sex == "Female" and age > 55 and fbs == "Yes":
            high_risk_patterns += 1
            adjusted_prob = min(adjusted_prob * 1.2, 0.90)
            pattern_details.append("Diabetic female >55 years")

    if high_risk_patterns > 0:
        if high_risk_patterns >= 2:
            clinical_notes.append(f"🚨 HIGH-RISK PATTERN: {high_risk_patterns} major risk combinations detected")
        else:
            clinical_notes.append(f"⚠️ Risk pattern: {high_risk_patterns} risk combination detected")

        for detail in pattern_details:
            clinical_notes.append(f"   • {detail}")

    probability_change = abs(adjusted_prob - raw_probability)

    if probability_change > 0.05 or clinical_notes:
        final_note = " | ".join(clinical_notes) if clinical_notes else None
        return adjusted_prob, final_note

    return raw_probability, None

#===================Load data and models====================================
def load_data():
    try:
        # Try multiple possible paths
        possible_paths = [
            'data/heart_disease_processed.csv',
            '../data/heart_disease_processed.csv',
            '../../data/heart_disease_processed.csv',
            'Heart_Disease_Project/data/heart_disease_processed.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        st.error(f"Data file not found. Checked: {possible_paths}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def apply_theme_css(dark_mode):
    """Apply comprehensive theme CSS"""
    if dark_mode:
        st.markdown("""
        <style>
            /* Dark mode styles */
            .main, .block-container, .stApp {
                background-color: #0E1117 !important;
                color: #FAFAFA !important;
            }
            
            .card {
                background-color: #1e1e1e !important;
                color: #FAFAFA !important;
                border-left: 4px solid #2E86AB !important;
            }
            
            .card h1, .card h2, .card h3, .card h4, .card h5, .card h6 {
                color: #2E86AB !important;
            }
            
            .card p, .card li, .card span {
                color: #FAFAFA !important;
            }
            
            .viz-container {
                background-color: #262730 !important;
                color: #FAFAFA !important;
            }
            
            .clinical-note {
                background: linear-gradient(135deg, #444 0%, #333 100%) !important;
                border-left: 5px solid #ffc107 !important;
                color: #FAFAFA !important;
            }
            
            .high-risk {
                background: linear-gradient(135deg, #660000 0%, #990000 100%) !important;
                border-left: 5px solid #ff4b4b !important;
                color: #FAFAFA !important;
            }
            
            .low-risk {
                background: linear-gradient(135deg, #003300 0%, #006600 100%) !important;
                border-left: 5px solid #00cc00 !important;
                color: #FAFAFA !important;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%) !important;
                color: white !important;
            }
            
            .nav-button {
                background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%) !important;
                color: #FAFAFA !important;
            }
            
            .nav-button.active {
                background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%) !important;
            }
            
            /* Ensure all text in main content is visible in dark mode */
            .main .block-container {
                color: #FAFAFA !important;
            }
            
            .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
                color: #2E86AB !important;
            }
            
            .main p, .main li, .main span, .main div {
                color: #FAFAFA !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
<style>
    /* Light mode styles */
    .main, .block-container, .stApp {
            /* Light mode styles */
            .main, .block-container, .stApp {
                background-color: #f8f9fa !important;
                color: #222222 !important;
            }
    div[data-testid="stRadio"], div[data-testid="stRadio"] * {
        color: #222222 !important;
        -webkit-text-fill-color: #222222 !important;
    }

    /* Fix other form component labels too (extra safety) */
    .stRadio label, .stCheckbox label, .stSelectbox label,
    .stSlider label, .stTextInput label, .stNumberInput label {
        color: #222222 !important;
        -webkit-text-fill-color: #222222 !important;
    }

    .card {
        background-color: #ffffff !important;
        color: #222222 !important;
    }
    .card h1, .card h2, .card h3, .card h4, .card h5, .card h6 {
        color: #2E86AB !important;
    }
    .card p, .card li, .card span {
        color: #444444 !important;
    }
    .viz-container {
        background-color: #ffffff !important;
        color: #222222 !important;
    }
    /* Ensure all text in main content is visible in light mode */
    .main .block-container {
        color: #222222 !important;
    }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #2E86AB !important;
    }
    .main p, .main li, .main span, .main div {
        color: #444444 !important;
    }
    /* Fix Streamlit component colors */
    .stRadio label, .stCheckbox label, .stSelectbox label {
        color: #222222 !important;
    }
    .stSlider label {
        color: #222222 !important;
    }
    .stTextInput label, .stNumberInput label {
        color: #222222 !important;
    }
</style>
""", unsafe_allow_html=True)

#==================================================Custom navigation component
def create_navigation():
    """Create custom navigation buttons"""

    nav_items = [
        {"icon": "🏠", "label": "Home", "key": "home"},
        {"icon": "🔮", "label": "Prediction", "key": "prediction"},
        {"icon": "📊", "label": "Analysis", "key": "analysis"},
        {"icon": "🤖", "label": "Insights", "key": "insights"}
    ]
    
    for item in nav_items:
        is_active = st.session_state.get('current_page', 'home') == item['key']
        button_class = "nav-button active" if is_active else "nav-button"
        
        if st.button(f"{item['icon']} {item['label']}", key=item['key']):
            st.session_state['current_page'] = item['key']
            st.rerun()

def main():

    if 'dark_mode' not in st.session_state:
        st.session_state['dark_mode'] = False
    if 'risk_result' not in st.session_state:
        st.session_state['risk_result'] = None
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'

    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Settings")
        st.session_state['dark_mode'] = st.checkbox("🌙 Dark Mode", value=st.session_state['dark_mode'])
        apply_theme_css(st.session_state['dark_mode'])
        
        st.markdown("### 🧭 Navigation")
        create_navigation()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app uses machine learning to assess heart disease risk based on clinical parameters.
        
        **Disclaimer:** This tool is for educational purposes only. Always consult healthcare professionals for medical diagnoses.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Load data and model
    df = load_data()
    model, scaler = load_model()

    # Main content area
    st.markdown('<h1 class="main-header">❤️ CardioPredict Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Advanced Heart Disease Risk Assessment with Clinical Intelligence</p>', unsafe_allow_html=True)

    # Page routing
    current_page = st.session_state.get('current_page', 'home')
    
    if current_page == 'home':
        show_home_page()
    elif current_page == 'prediction':
        show_prediction_page(model, scaler)
    elif current_page == 'analysis':
        show_advanced_analysis_page(df)
    elif current_page == 'insights':
        show_model_insights_page(df, model)


#===============================================Home_page=========================
def show_home_page():
    st.markdown("""
    <div class="card">
    <h2>Welcome to CardioPredict Pro</h2>
    <p>This advanced clinical tool predicts the likelihood of heart disease using machine learning 
    enhanced with clinical validation rules for improved patient safety.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>🎯 Accurate Predictions</h3>
        <p>Machine learning model with 85-90% accuracy based on clinical data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>🛡️ Clinical Safety</h3>
        <p>Built-in clinical validation for rare cases and edge scenarios.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
        <h3>📈 Data Insights</h3>
        <p>Comprehensive analysis and visualization of risk factors.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics row
    st.markdown('<div class="sub-header">System Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>Model Accuracy</h3><h2>85-90%</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>Training Data</h3><h2>300+</h2><p>Patient records</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>Features</h3><h2>13</h2><p>Clinical parameters</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>Safety Checks</h3><h2>5+</h2><p>Clinical rules</p></div>', unsafe_allow_html=True)
    
    # How to use section
    st.markdown('<div class="sub-header">How to Use</div>', unsafe_allow_html=True)
    
    steps_col1, steps_col2, steps_col3 = st.columns(3)
    
    with steps_col1:
        st.markdown("""
        <div class="card">
        <h3>1. Navigate to Prediction</h3>
        <p>Use the sidebar to go to the Prediction page.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col2:
        st.markdown("""
        <div class="card">
        <h3>2. Enter Patient Data</h3>
        <p>Fill in the clinical parameters using the intuitive form.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col3:
        st.markdown("""
        <div class="card">
        <h3>3. View Results</h3>
        <p>Get immediate risk assessment with clinical context.</p>
        </div>
        """, unsafe_allow_html=True)


#==================================================prediction_page=========================
def show_prediction_page(model, scaler):
    st.markdown('<div class="sub-header">Patient Assessment</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please check if model files exist.")
        return

    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("prediction_form"):
            st.markdown("""
            <div class="card">
            <h3>Patient Information</h3>
            """, unsafe_allow_html=True)
            demo_tab, clinical_tab, cardiac_tab = st.tabs(["Demographics", "Clinical Metrics", "Cardiac Parameters"])
            
            with demo_tab:
                col1a, col2a = st.columns(2)
                with col1a:
                    age = st.slider("Age", 20, 100, 50, help="Patient age in years")
                    sex = st.radio("Sex", ["Male", "Female"])
                    cp = st.selectbox("Chest Pain Type",
                                      ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"],
                                      help="Type of chest pain experienced")
                
                with col2a:
                    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
                    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
                    fbs = st.radio("Fasting Blood Sugar > 120mg/dl", ["No", "Yes"])
            
            with clinical_tab:
                col1b, col2b = st.columns(2)
                with col1b:
                    restecg = st.selectbox("Resting ECG Results",
                                           ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
                    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150,
                                       help="Maximum heart rate during exercise")
                
                with col2b:
                    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
                    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
            
            with cardiac_tab:
                col1c, col2c = st.columns(2)
                with col1c:
                    slope = st.slider("Slope of Peak Exercise ST Segment", 1, 3, 2,
                                     help="1: Upsloping, 2: Flat, 3: Downsloping")
                    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
                
                with col2c:
                    thal = st.selectbox("Thalassemia",
                                        ["Normal", "Fixed Defect", "Reversible Defect"],
                                        help="Thalassemia test results")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            submitted = st.form_submit_button("🔍 Assess Cardiovascular Risk", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>Clinical Guidance</h3>
        <p><strong>Typical Angina:</strong> Substernal chest discomfort provoked by exertion or stress</p>
        <p><strong>ST Depression:</strong> Indicator of myocardial ischemia</p>
        <p><strong>Thalassemia:</strong> Blood disorder that can affect heart function</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick reference values
        st.markdown("""
        <div class="card">
        <h3>Normal Ranges</h3>
        <ul>
        <li>BP: <120/80 mm Hg</li>
        <li>Cholesterol: <200 mg/dL</li>
        <li>Fasting Glucose: <100 mg/dL</li>
        <li>Max HR: ~220 - age</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        
        with st.spinner('Analyzing patient data...'):
            
            import time
            time.sleep(1)
            
            # Convert inputs to model format
            sex_num = 1 if sex == "Male" else 0
            fbs_num = 1 if fbs == "Yes" else 0
            exang_num = 1 if exang == "Yes" else 0

            cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal": 3, "Asymptomatic": 4}
            restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}

            features = np.array([[age, sex_num, cp_map[cp], trestbps, chol, fbs_num,
                                  restecg_map[restecg], thalach, exang_num, oldpeak, slope, ca, thal_map[thal]]])

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            raw_probability = model.predict_proba(features_scaled)[0][1]

            final_probability, clinical_note = clinical_override(
                age=age, thalach=thalach, raw_probability=raw_probability,
                sex=sex, cp=cp, exang=exang, oldpeak=oldpeak,
                thal=thal, fbs=fbs, trestbps=trestbps, chol=chol
            )

            risk_percentage = final_probability * 100
            raw_risk_percentage = raw_probability * 100
            
            st.session_state['risk_result'] = {
                'prediction': prediction,
                'final_probability': final_probability,
                'risk_percentage': risk_percentage,
                'raw_risk_percentage': raw_risk_percentage,
                'clinical_note': clinical_note,
                'age': age,
                'thalach': thalach
            }

        display_prediction_results()

def display_prediction_results():
    if st.session_state['risk_result'] is None:
        return
        
    result = st.session_state['risk_result']
    prediction = result['prediction']
    risk_percentage = result['risk_percentage']
    clinical_note = result['clinical_note']

    st.markdown('<div class="sub-header">Assessment Results</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'lightcoral'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}
            }
        ))
        
        fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:

        if prediction == 1 and risk_percentage > 50:
            st.markdown('<div class="high-risk">', unsafe_allow_html=True)
            st.error(f"🚨 HIGH CARDIOVASCULAR RISK - {risk_percentage:.1f}% probability")
            st.markdown("""
            **Clinical Recommendation:** 
            - Consult cardiologist immediately
            - Consider further diagnostic testing
            - Implement risk factor modification
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="low-risk">', unsafe_allow_html=True)
            st.success(f"✅ LOW CARDIOVASCULAR RISK - {risk_percentage:.1f}% probability")
            st.markdown("""
            **Clinical Recommendation:** 
            - Maintain healthy lifestyle
            - Regular follow-up as appropriate
            - Continue preventive measures
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        if clinical_note:
            st.markdown(f'<div class="clinical-note">{clinical_note}</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Clinical Context", "Input Summary", "Risk Factors"])
    
    with tab1:
        st.markdown("""
        <div class="card">
        <h3>Clinical Interpretation</h3>
        """, unsafe_allow_html=True)
        
        result = st.session_state['risk_result']
        age = result['age']
        thalach = result['thalach']
        predicted_max_hr = 220 - age
        hr_percentage = (thalach / predicted_max_hr) * 100
        
        st.write(f"**Age:** {age} years")
        st.write(f"**Expected Max HR:** {predicted_max_hr} bpm")
        st.write(f"**Achieved HR:** {thalach} bpm ({hr_percentage:.1f}% of expected)")
        
        if hr_percentage > 85:
            st.write("**Exercise Capacity:** Normal")
        elif hr_percentage > 70:
            st.write("**Exercise Capacity:** Below average")
        else:
            st.write("**Exercise Capacity:** Low")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="card">
        <h3>Patient Parameters</h3>
        <p>All input values have been recorded and factored into the risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        # Risk factor analysis
        st.markdown("""
        <div class="card">
        <h3>Modifiable Risk Factors</h3>
        <ul>
        <li>Blood pressure management</li>
        <li>Cholesterol control</li>
        <li>Blood glucose regulation</li>
        <li>Weight management</li>
        <li>Physical activity</li>
        <li>Smoking cessation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


#==================================================analysis_page=========================
def show_advanced_analysis_page(df):
    st.header("📊 Advanced Data Analysis")

    if df is None:
        st.error("Data not loaded. Please run the data preprocessing notebook first.")
        return

    # Add clinical context about data limitations
    st.info("""
    **Dataset Context:** This model was trained on patients aged 29-77 years. 
    Predictions for edge cases (very young/old patients with unusual profiles) include clinical safety adjustments.
    """)

    analysis_type = st.selectbox("Choose Analysis Type", [
        "Basic Distributions",
        "PCA Analysis",
        "Feature Correlations",
        "Clustering Insights",
        "Comparative Analysis"
    ])

    if analysis_type == "Basic Distributions":
        show_basic_distributions(df)
    elif analysis_type == "PCA Analysis":
        show_pca_analysis(df)
    elif analysis_type == "Feature Correlations":
        show_feature_correlations(df)
    elif analysis_type == "Clustering Insights":
        show_clustering_insights(df)
    elif analysis_type == "Comparative Analysis":
        show_comparative_analysis(df)

def show_basic_distributions(df):
    st.subheader("📈 Basic Distributions")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        df['target'].value_counts().plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
        ax.set_title('Heart Disease Distribution')
        ax.set_xlabel('Heart Disease (0 = No, 1 = Yes)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots(figsize=(8, 4))
        for target_val, color, label in [(0, 'blue', 'Healthy'), (1, 'red', 'Heart Disease')]:
            data = df[df['target'] == target_val]['age']
            ax.hist(data, alpha=0.7, color=color, label=label, bins=15)
        ax.legend()
        ax.set_title('Age Distribution by Heart Disease')
        ax.set_xlabel('Age')
        st.pyplot(fig)

    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select feature",
                                   ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for target_val in [0, 1]:
        data = df[df['target'] == target_val][selected_feature]
        ax1.hist(data, alpha=0.7, label=f'Disease = {target_val}')
    ax1.legend()
    ax1.set_title(f'{selected_feature} Distribution')


    df.boxplot(column=selected_feature, by='target', ax=ax2)
    ax2.set_title(f'{selected_feature} by Heart Disease')

    st.pyplot(fig)

def show_pca_analysis(df):
    st.subheader("🔍 PCA Analysis")
    X = df.drop(['target', 'target_original'], axis=1, errors='ignore')
    y = df['target']

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('PCA: Heart Disease Visualization')
    plt.colorbar(scatter, label='Heart Disease')
    st.pyplot(fig)

    st.subheader("PCA Explained Variance")
    pca_full = PCA().fit(X)
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-')
    ax.axhline(y=0.85, color='g', linestyle='--', label='85% Variance')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.info(f"**PCA Insights:** First 2 components explain {pca.explained_variance_ratio_.sum():.2%} of variance")

def show_feature_correlations(df):
    st.subheader("📊 Feature Correlations")

    # Correlation heatmapppp
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['target_original']]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)

    st.subheader("Top Features Correlated with Heart Disease")
    target_correlations = df[numeric_cols].corr()['target'].abs().sort_values(ascending=False)
    target_correlations = target_correlations[target_correlations.index != 'target']

    fig, ax = plt.subplots(figsize=(10, 6))
    target_correlations.head(10).plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title('Top Features Correlated with Heart Disease')
    ax.set_xlabel('Absolute Correlation with Target')
    st.pyplot(fig)

def show_clustering_insights(df):
    st.subheader("👥 Clustering Insights")

    X = df.drop(['target', 'target_original'], axis=1, errors='ignore')

    # Simple K-means clustering
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler_local = StandardScaler()
    X_scaled = scaler_local.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # True labels
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['target'], cmap='viridis', alpha=0.7)
    ax1.set_title('True Labels (Heart Disease)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1)

    # Clusters
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='Set2', alpha=0.7)
    ax2.set_title('K-means Clusters (k=3)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    plt.colorbar(scatter2, ax=ax2)

    st.pyplot(fig)

    # Cluster analysis
    st.subheader("Cluster Characteristics")
    df_cluster = df.copy()
    df_cluster['cluster'] = clusters

    cluster_summary = df_cluster.groupby('cluster').mean()
    st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))

def show_comparative_analysis(df):
    st.subheader("📋 Comparative Analysis")
    healthy = df[df['target'] == 0]
    disease = df[df['target'] == 1]
    features_to_compare = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for i, feature in enumerate(features_to_compare):
        if i < len(axes):
            
            data_to_plot = [healthy[feature], disease[feature]]
            axes[i].boxplot(data_to_plot, labels=['Healthy', 'Heart Disease'])
            axes[i].set_title(f'{feature} Comparison')
            axes[i].set_ylabel(feature)

    # Remove empty subplots
    for i in range(len(features_to_compare), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Statistical Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Healthy Patients Summary:**")
        st.dataframe(healthy[features_to_compare].describe())

    with col2:
        st.write("**Heart Disease Patients Summary:**")
        st.dataframe(disease[features_to_compare].describe())

def show_model_insights_page(df, model):
    st.header("🤖 Model Insights")

    if model is None or df is None:
        st.error("Model or data not loaded properly.")
        return


    st.info("""
    **Model Safety Features:** Includes clinical override rules for rare cases:
    - Young patients (<40) with very low exercise capacity
    - Elderly patients with unusually high exercise capacity
    - Transparent display of raw vs adjusted predictions
    """)
    show_feature_importance(df, model)

def show_feature_importance(df, model):
    st.subheader("🎯 Feature Importance")


    if hasattr(model, 'feature_importances_'):
        feature_names = df.drop(['target', 'target_original'], axis=1, errors='ignore').columns
        importance = model.feature_importances_


        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)


        fig, ax = plt.subplots(figsize=(10, 8))
        importance_df.plot(kind='barh', x='feature', y='importance', ax=ax, color='lightgreen')
        ax.set_title('Feature Importance Ranking')
        ax.set_xlabel('Importance Score')
        st.pyplot(fig)
        st.subheader("Feature Importance Scores")
        st.dataframe(importance_df.sort_values('importance', ascending=False))

        st.subheader("📋 Key Insights")
        top_features = importance_df.nlargest(3, 'importance')['feature'].tolist()
        st.write(f"**Most important features:** {', '.join(top_features)}")
        st.write("These features have the strongest influence on heart disease prediction.")

        if 'thalach' in feature_names:
            thalach_importance = importance_df[importance_df['feature'] == 'thalach']['importance'].iloc[0]
            st.write(f"**Max Heart Rate (thalach) importance:** {thalach_importance:.3f}")
            st.write("Note: Heart rate predictions include clinical adjustments for age-appropriate ranges.")

    else:
        st.info("Feature importance not available for this model type.")

        # Show correlation-based importance as fallback
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['target_original']]
        correlations = df[numeric_cols].corr()['target'].abs().sort_values(ascending=False)
        correlations = correlations[correlations.index != 'target']

        fig, ax = plt.subplots(figsize=(10, 6))
        correlations.head(10).plot(kind='barh', ax=ax, color='lightblue')
        ax.set_title('Feature Correlation with Target')
        ax.set_xlabel('Absolute Correlation')
        st.pyplot(fig)

        st.subheader("📋 Key Insights")
        top_correlated = correlations.head(3).index.tolist()
        st.write(f"**Most correlated features:** {', '.join(top_correlated)}")
        st.write("These features show the strongest statistical relationship with heart disease.")


if __name__ == "__main__":

    main()