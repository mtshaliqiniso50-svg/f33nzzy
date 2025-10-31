import streamlit as st
import numpy as np
import pandas as pd
import base64 # Esetshenziselwa ukufaka i-CSS yangokwezifiso

# --- 1. CONFIGURATION AND STYLING (Deep Ocean Theme) ---

# Ukusetha i-page configuration ebanzi
st.set_page_config(
    page_title="Telco Churn Risk Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# I-CSS yangokwezifiso yokwenza i-UI ibukeke ihlanzekile
def load_css():
    """Custom CSS for a cleaner, modern look with Deep Ocean feel."""
    # Use base64 to embed CSS for better consistency
    css = """
    <style>
    /* Main Streamlit container adjustments */
    .stApp {
        background-color: #1a202c; /* Dark blue/gray background */
        color: #e2e8f0; /* Light text */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4dc0b5; /* Teal/Cyan accent color for titles */
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748; /* Darker tab background */
        color: #a0aec0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4a5568; /* Slightly lighter on hover */
    }
    .stTabs [aria-selected="true"] {
        background-color: #4dc0b5; /* Active tab accent color */
        color: #1a202c !important; /* Dark text on active tab */
        border-bottom: 3px solid #4dc0b5; 
        font-weight: bold;
    }
    /* Card/Block styling */
    .stAlert, .stProgress, .stProgress > div, .stProgress > div > div {
        border-radius: 12px;
    }
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        padding: 20px;
        border-radius: 12px;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2d3748; /* Slightly lighter sidebar */
        color: #e2e8f0;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css() # Call the CSS loader function

# --- 2. MODEL COEFFICIENTS (Simulated Logistic Regression Logic) ---
# Sisebenzisa i-simulated coefficients njengoba kwadingeka, kugxilwe kwi-Recall.
CHURN_COEFFICIENTS = {
    'Contract_Month-to-month': 1.5,
    'InternetService_Fiber optic': 1.1,
    'PaymentMethod_Electronic check': 0.6,
    
    'Contract_Two year': -1.2,
    'Tenure_High': -1.0, 
    
    'Tenure_Low': 0.8,
    'MonthlyCharges_High': 0.4,
    
    'Intercept': -1.8 
}

# --- 3. FEATURE IMPORTANCE DISPLAY (For Insights) ---
FEATURE_IMPORTANCE = [
    {"name": "I-Contract (Month-to-month)", "effect": "Uphawu Olukhulu Lwe-Churn", "color": "🔴"},
    {"name": "I-Tenure Ephansi (< 12 months)", "effect": "Ingozi Enkulu Yokuhamba", "color": "🔴"},
    {"name": "I-Fiber Optic Internet", "effect": "Kudingeka Ukubhekwa", "color": "🟠"},
    {"name": "I-Electronic Check Payment", "effect": "Ingozi Engeziwe", "color": "🟠"},
    {"name": "I-Two-Year Contract", "effect": "Isici Sokugcina Ikhasimende", "color": "🟢"},
]

# --- 4. CORE FUNCTIONS ---

def sigmoid(x):
    """Converts a linear score to a probability (0 to 1)"""
    return 1 / (1 + np.exp(-x))

def predict_churn(inputs):
    """Simulates the Logistic Regression prediction based on inputs."""
    linear_score = CHURN_COEFFICIENTS['Intercept']

    # Apply feature effects
    if inputs['contract'] == 'Month-to-month':
        linear_score += CHURN_COEFFICIENTS['Contract_Month-to-month']
    elif inputs['contract'] == 'Two year':
        linear_score += CHURN_COEFFICIENTS['Contract_Two year']

    if inputs['tenure'] <= 12:
        linear_score += CHURN_COEFFICIENTS['Tenure_Low'] 
    elif inputs['tenure'] >= 60:
        linear_score += CHURN_COEFFICIENTS['Tenure_High'] 

    if inputs['internetService'] == 'Fiber optic':
        linear_score += CHURN_COEFFICIENTS['InternetService_Fiber optic']

    if inputs['monthlyCharges'] >= 85:
        linear_score += CHURN_COEFFICIENTS['MonthlyCharges_High']
    
    if inputs['paymentMethod'] == 'Electronic check':
        linear_score += CHURN_COEFFICIENTS['PaymentMethod_Electronic check']
    
    return sigmoid(linear_score)

# --- 5. SIDEBAR: USER INPUT FORM ---

with st.sidebar:
    st.image("https://placehold.co/100x100/4dc0b5/1a202c?text=AI", use_column_width=False)
    st.markdown("---")
    st.header("1. I-Model Input")
    st.caption("Faka imininingwane yekhasimende ukuze ubale ingozi ye-Churn.")
    
    contract = st.selectbox(
        "I-Contract Type",
        ('Month-to-month', 'One year', 'Two year'),
        index=0,
    )
    tenure = st.slider(
        "Tenure (Inyanga)",
        min_value=1, max_value=72, value=12, step=1,
    )
    internetService = st.selectbox(
        "I-Internet Service",
        ('DSL', 'Fiber optic', 'No'),
        index=1,
    )
    monthlyCharges = st.number_input(
        "Monthly Charges ($)",
        min_value=18.25, max_value=118.75, value=75.00, step=0.01,
    )
    paymentMethod = st.selectbox(
        "I-Payment Method",
        ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'),
        index=0,
    )

    inputs = {
        'contract': contract,
        'tenure': tenure,
        'internetService': internetService,
        'monthlyCharges': monthlyCharges,
        'paymentMethod': paymentMethod,
    }

# Run Prediction
probability = predict_churn(inputs)
percentage = round(probability * 100)

# --- 6. MAIN CONTENT: TABS ---

st.title("Telco Customer Churn Risk Dashboard")
st.markdown("I-Model esetshenziswayo: **Logistic Regression** (I-Recall Ephakeme Kakhulu Yokubamba Amakhasimende)")
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Churn Predictor & Insights", "🏆 Model Performance"])

# --- TAB 1: PREDICTOR ---
with tab1:
    st.header("Churn Risk Assessment")
    st.caption("Bheka umphumela we-prediction ngesikhathi sangempela bese uthola izincomo zokulwa ne-churn.")
    
    colA, colB = st.columns([1, 2])

    # Column A: Prediction Output
    with colA:
        st.subheader("2. I-Churn Probability")

        if percentage >= 70:
            risk_text = 'I-CRITICAL RISK!'
            delta_color = 'inverse'
            message = 'Ingozi iphezulu kakhulu. Qala i-campaign yokubamba ngokushesha!'
            st.error(f"⚠️ {message}")
        elif percentage >= 40:
            risk_text = 'I-HIGH RISK'
            delta_color = 'normal'
            message = 'Ingozi iphakathi. Kuningi okungase kushintshe ukuze kulungiseke.'
            st.warning(f"🔔 {message}")
        else:
            risk_text = 'I-LOW RISK'
            delta_color = 'off'
            message = 'Ikhasimende lizinzile. Qhubeka nokubheka ukuze kuqhubeke kube kuhle.'
            st.success(f"✅ {message}")

        # Display Gauge Simulation (Metric)
        st.metric(
            label="I-Churn Probability Score", 
            value=f"{percentage}%", 
            delta=risk_text,
            delta_color=delta_color
        )
        
        st.progress(probability)
        st.markdown(f"**I-Threshold:** Ikhasimende ligcinwa uma i-probability < 50%.")

    # Column B: Feature Insights
    with colB:
        st.subheader("3. I-Key Churn Drivers")
        st.markdown("Amacoefficient e-model akhombisa ukuthi yiziphi izinto ezibangela ingozi:")
        
        # Create a table/dataframe for better visualization of importance
        df_importance = pd.DataFrame(FEATURE_IMPORTANCE)
        df_importance['Impact'] = df_importance.apply(
            lambda row: f"{row['color']} {row['effect']}", axis=1
        )
        df_importance = df_importance[['name', 'Impact']]
        df_importance.columns = ['I-Feature', 'Umthelela We-Risk']
        
        st.dataframe(
            df_importance, 
            hide_index=True,
            use_container_width=True
        )

# --- TAB 2: MODEL PERFORMANCE ---
with tab2:
    st.header("Model Evaluation Summary")
    st.caption("Ukuqhathaniswa kwe-Random Forest (RF) ne-Logistic Regression (LR) ku-Testing Data.")
    
    # Static Data for performance comparison
    performance_data = {
        'Metric': ['Overall Accuracy', 'Recall (Churn: Yes)', 'Precision (Churn: Yes)', 'Model Size/Complexity'],
        'Random Forest (RF)': ['79%', '71%', '62%', 'High (Slow Deployment)'],
        'Logistic Regression (LR) - Yethu': ['75%', '76%', '61%', 'Low (Fast Deployment)'],
    }
    df_performance = pd.DataFrame(performance_data)
    
    st.table(df_performance)
    
    st.markdown("### Isiphetho Esibalulekile")
    st.info("""
        **I-Decision:** Sikhethe i-Logistic Regression ngenxa ye-**Recall** yayo engu-**76%**.
        Noma i-Random Forest ine-Accuracy ephezulu (79%), i-LR iyakwazi ukuthola amakhasimende amaningi azohamba. 
        Kwezokubamba amakhasimende (Retention), ukuthola ikhasimende elisengozini kubaluleke kakhulu kunokubikezela okulungile kwamakhasimende azinzile.
    """)
    
st.markdown("---")
st.caption("I-Deep Ocean Dashboard | Yenziwe nge-Streamlit | Powered by Gemini")


