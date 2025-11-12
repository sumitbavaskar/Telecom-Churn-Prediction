import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📱",
    layout="wide"
)

# Title and description
st.title("📱 Telecom Churn Prediction App")
st.markdown("""This app predicts whether a telecom customer will churn based on their usage patterns and service characteristics.""")

# Load the trained model
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load data for reference
@st.cache_data
def load_data():
    df = pd.read_csv('telecom_churn_cleaned.csv')
    return df

try:
    model = load_model()
    df = load_data()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Sidebar for input
st.sidebar.header("Customer Information")
st.sidebar.markdown("Enter customer details below:")

# Input fields
account_length = st.sidebar.number_input("Account Length (days)", min_value=1, max_value=243, value=100)
international_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"])
vmail_plan = st.sidebar.selectbox("Voice Mail Plan", ["No", "Yes"])
vmail_message = st.sidebar.number_input("Voice Mail Messages", min_value=0, max_value=51, value=0)

st.sidebar.subheader("Usage Statistics")
day_mins = st.sidebar.number_input("Day Minutes", min_value=0.0, max_value=400.0, value=180.0)
day_calls = st.sidebar.number_input("Day Calls", min_value=0, max_value=165, value=100)
eve_mins = st.sidebar.number_input("Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
eve_calls = st.sidebar.number_input("Evening Calls", min_value=0, max_value=175, value=100)
night_mins = st.sidebar.number_input("Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
night_calls = st.sidebar.number_input("Night Calls", min_value=0, max_value=175, value=100)
intl_mins = st.sidebar.number_input("International Minutes", min_value=0.0, max_value=20.0, value=10.0)
intl_calls = st.sidebar.number_input("International Calls", min_value=0, max_value=20, value=4)
custserv_calls = st.sidebar.number_input("Customer Service Calls", min_value=0, max_value=9, value=1)

# State selection
state = st.sidebar.selectbox("State", sorted(df.filter(like='State_').columns.str.replace('State_', '').tolist()))
area_code = st.sidebar.selectbox("Area Code", ["408", "415", "510"])

# Predict button
if st.sidebar.button("🔮 Predict Churn", type="primary"):
    # Convert inputs
    intl_plan_num = 1 if international_plan == "Yes" else 0
    vmail_plan_num = 1 if vmail_plan == "Yes" else 0
    
    # Calculate engineered features
    total_mins = day_mins + eve_mins + night_mins + intl_mins
    total_calls = day_calls + eve_calls + night_calls + intl_calls
    
    avg_day_duration = day_mins / max(day_calls, 1)
    avg_eve_duration = eve_mins / max(eve_calls, 1)
    avg_night_duration = night_mins / max(night_calls, 1)
    avg_intl_duration = intl_mins / max(intl_calls, 1)
    avg_call_duration = total_mins / max(total_calls, 1)
    
    custserv_ratio = custserv_calls / max(total_calls, 1)
    intl_mins_ratio = intl_mins / max(total_mins, 1)
    intl_calls_ratio = intl_calls / max(total_calls, 1)
    vmail_msg_ratio = vmail_message / 52
    
    high_custserv = 1 if custserv_calls > df['CustServ Calls'].quantile(0.90) else 0
    high_total_mins = 1 if total_mins > 662 else 0
    high_total_calls = 1 if total_calls > 359 else 0
    
    # Tenure category
    if account_length <= 50:
        tenure_cat = 0
    elif account_length <= 100:
        tenure_cat = 1
    elif account_length <= 150:
        tenure_cat = 2
    elif account_length <= 200:
        tenure_cat = 3
    else:
        tenure_cat = 4
    
    # Create feature vector (80 features)
    features = [
        account_length, intl_plan_num, vmail_plan_num, vmail_message,
        day_mins, day_calls, eve_mins, eve_calls,
        night_mins, night_calls, intl_mins, intl_calls,
        custserv_calls, total_mins, total_calls,
        avg_day_duration, avg_eve_duration, avg_night_duration,
        avg_intl_duration, avg_call_duration,
        custserv_ratio, intl_mins_ratio, intl_calls_ratio, vmail_msg_ratio,
        high_custserv, high_total_mins, high_total_calls, tenure_cat
    ]
    
    # Add state one-hot encoding (50 features for 50 states after drop_first)
    all_states = sorted(df.filter(like='State_').columns.str.replace('State_', '').tolist())
    for s in all_states:
        features.append(1 if s == state else 0)
    
    # Add area code one-hot encoding (2 features after drop_first)
    features.append(1 if area_code == "415" else 0)
    features.append(1 if area_code == "510" else 0)
    
    # Create dataframe for prediction
    feature_names = (
        ['Account Length', 'International Plan', 'VMail Plan', 'VMail Message',
         'Day Mins', 'Day Calls', 'Eve Mins', 'Eve Calls',
         'Night Mins', 'Night Calls', 'International Mins', 'International Calls',
         'CustServ Calls', 'Total Mins', 'Total Calls',
         'Avg Day Call Duration', 'Avg Eve Call Duration', 'Avg Night Call Duration',
         'Avg International Call Duration', 'Avg Call Duration',
         'CustServ Calls Ratio', 'International Mins Ratio', 'International Calls Ratio',
         'VMail Message Ratio', 'High CustServ Calls', 'High Total Mins',
         'High Total Calls', 'Tenure Category Numeric'] +
        [f'State_{s}' for s in all_states] +
        ['AreaCode_415', 'AreaCode_510']
    )
    
    input_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale the features
    scaler = StandardScaler()
    # Fit scaler on training data statistics (approximate)
    scaler.fit(df.drop('Churn', axis=1))
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.header("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Prediction", "YES - Will Churn" if prediction == 1 else "NO - Will Stay", 
                  delta="High Risk" if prediction == 1 else "Low Risk",
                  delta_color="inverse")
    
    with col2:
        st.metric("Churn Probability", f"{prediction_proba[1]*100:.2f}%")
    
    with col3:
        st.metric("Stay Probability", f"{prediction_proba[0]*100:.2f}%")
    
    # Risk level
    if prediction_proba[1] > 0.7:
        risk_level = "🔴 HIGH RISK"
        risk_color = "red"
    elif prediction_proba[1] > 0.4:
        risk_level = "🟡 MEDIUM RISK"
        risk_color = "orange"
    else:
        risk_level = "🟢 LOW RISK"
        risk_color = "green"
    
    st.markdown(f"### Risk Level: {risk_level}")
    
    # Recommendations
    st.subheader("Recommendations")
    if prediction == 1:
        st.warning("⚠️ This customer is likely to churn. Consider the following actions:")
        recommendations = []
        
        if custserv_calls >= 4:
            recommendations.append("- High customer service calls detected. Investigate and resolve customer issues immediately.")
        if intl_plan_num == 1 and intl_mins > 15:
            recommendations.append("- Customer has high international usage. Offer better international plan benefits.")
        if total_mins < 400:
            recommendations.append("- Low usage detected. Offer promotional minutes or engage with personalized offers.")
        if vmail_plan_num == 0:
            recommendations.append("- Customer doesn't have voicemail plan. Offer value-added services.")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.markdown("- Reach out proactively with retention offers")
            st.markdown("- Conduct satisfaction survey to understand pain points")
    else:
        st.success("✅ This customer is likely to stay. Continue providing excellent service!")
        st.markdown("- Maintain current service quality")
        st.markdown("- Consider upselling opportunities")
        st.markdown("- Encourage referrals through loyalty programs")

# Display sample data
with st.expander("📊 View Sample Data"):
    st.dataframe(df.head(10))

# Model information
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    ### Model Information
    - **Algorithm**: Random Forest Classifier
    - **Accuracy**: 93.29%
    - **Features**: 80 features including usage patterns, service plans, and customer demographics
    - **Training Data**: 4,617 customer records
    
    ### Key Features:
    - Customer service call frequency
    - International plan usage
    - Total minutes and calls
    - Account tenure
    - Geographic location
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Telecom Churn Prediction Model")
