"""
Fraud Detection Streamlit App

A production-ready web application for predicting fraudulent financial transactions.
This app uses a trained Gradient Boosting Classifier model to classify transactions
as legitimate (0) or fraudulent (1).
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from typing import Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton > button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton > button:hover { background-color: #1d4ed8; }
    .result-legitimate {
        background-color: #d1fae5;
        color: #065f46;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
    }
    .result-fraud {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
    }
    .feature-input {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model, scaler, and feature columns."""
    try:
        project_dir = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(project_dir, 'fraud_detection_model.joblib')
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None, None

        model = joblib.load(model_path)

        scaler_path = os.path.join(project_dir, 'scaler.joblib')
        scaler = joblib.load(scaler_path)

        columns_path = os.path.join(project_dir, 'model_columns.joblib')
        columns = joblib.load(columns_path)

        st.sidebar.success("Model and preprocessing artifacts loaded successfully!")
        return model, scaler, columns

    except Exception as e:
        st.sidebar.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None


def create_feature_input_form(columns: list) -> Tuple[np.ndarray, Optional[dict]]:
    """Create Streamlit form for feature input."""
    st.header("Transaction Details")
    st.write("Enter the transaction details to check for potential fraud:")

    input_values = {}

    # Organize features
    financial_features = [c for c in columns if c not in
                          ['step'] and 'zero_balance' not in c and 'balance_diff' not in c and 'type_' not in c]
    derived_features = [c for c in columns if 'balance_diff' in c or 'zero_balance' in c]
    categorical_features = [c for c in columns if c.startswith('type_')]

    # Step input first
    st.subheader("Transaction Step")
    input_values['step'] = st.number_input(
        "Step Number", min_value=1, max_value=1000, value=1, step=1
    )

    # Transaction Type
    st.subheader("Transaction Type")
    type_options = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    selected_type = st.selectbox(
        "Select Transaction Type",
        options=type_options,
        index=type_options.index('TRANSFER') if 'type_TRANSFER' in columns else 0
    )

    # Financial inputs
    st.subheader("Financial Information")
    for feature in financial_features:
        if 'balance' in feature and 'diff' not in feature:
            input_values[feature] = st.number_input(
                feature.replace('_', ' ').title(),
                min_value=0.0, max_value=1e8, value=0.0, step=1.0, format="%.2f"
            )
        else:
            input_values[feature] = st.number_input(
                feature.replace('_', ' ').title(),
                min_value=-1e8, max_value=1e8, value=0.0, step=1.0
            )

    # Derived features
    st.subheader("Derived Features")
    for feature in derived_features:
        if 'is_zero' in feature:
            input_values[feature] = st.selectbox(
                feature.replace('_', ' ').title(),
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        else:
            input_values[feature] = st.number_input(
                feature.replace('_', ' ').title(),
                min_value=-1e8, max_value=1e8, value=0.0, step=1.0
            )

    # One-hot encoding for type
    for type_col in categorical_features:
        input_values[type_col] = 1 if type_col.replace('type_', '') == selected_type else 0

    return input_values, {'type_selected': selected_type, 'step': input_values['step']}


def validate_inputs(input_values: dict, columns: list) -> Tuple[bool, str]:
    """Validate user inputs."""
    missing_cols = [col for col in columns if col not in input_values]
    if missing_cols:
        return False, f"Missing input fields for: {', '.join(missing_cols)}"

    negative_fields = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                       'oldbalanceDest', 'newbalanceDest']
    for field in negative_fields:
        if field in input_values and input_values[field] < 0:
            return False, f"Field '{field}' cannot be negative"

    if 'step' in input_values and (input_values['step'] < 1 or input_values['step'] > 1000):
        return False, "Step number must be between 1 and 1000"

    return True, "Inputs are valid"


def prepare_features(input_values: dict, scaler, columns: list) -> Optional[np.ndarray]:
    """Prepare and scale features for prediction."""
    try:
        df = pd.DataFrame([input_values])
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return None
        df = df[columns]
        scaled_features = scaler.transform(df)
        return scaled_features
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        return None


def predict_fraud(model, features: np.ndarray) -> Tuple[int, Optional[float]]:
    """Make prediction using the model."""
    try:
        prediction = model.predict(features)[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
        else:
            confidence = None
        return prediction, confidence
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None


def display_results(prediction: int, confidence: Optional[float], input_data: dict):
    """Display prediction results."""
    st.markdown("## Prediction Results")

    if prediction == 0:
        st.markdown("""
        <div class="result-legitimate">
            <h3>✅ Legitimate Transaction</h3>
            <p>This transaction appears to be <strong>legitimate</strong>.</p>
            <p><strong>Model Decision:</strong> 0 (Not Fraud)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-fraud">
            <h3>🚨 Potential Fraud Detected</h3>
            <p>This transaction is <strong>flagged as potentially fraudulent</strong>.</p>
            <p><strong>Model Decision:</strong> 1 (Fraud)</p>
        </div>
        """, unsafe_allow_html=True)

    if confidence is not None:
        st.markdown(f"<p><strong>Confidence Level:</strong> {confidence:.2%}</p>", unsafe_allow_html=True)

    st.markdown("### Transaction Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Amount", f"${input_data.get('amount', 0):,.2f}")
        st.metric("Transaction Type", input_data.get('type_selected', 'Unknown'))
    with col2:
        st.metric("Step", input_data.get('step', 'N/A'))
        fraud_risk = "High" if prediction == 1 else "Low"
        st.metric("Risk Level", fraud_risk)

    with st.expander("Detailed Model Information"):
        st.write("**Model Used:** Gradient Boosting Classifier")
        st.write("**Features Analyzed:**", len(input_data) - 2)


def main():
    """Main Streamlit application."""
    st.title("🔍 Fraud Detection System")
    st.markdown("""
    This system uses machine learning to analyze financial transactions and identify
    potential fraudulent activity. Enter the transaction details below to get an
    instant prediction.

    **Note:** This tool is for demonstration purposes. Real fraud detection
    systems require additional analysis and human review.
    """)

    # Load model and artifacts
    with st.spinner("Loading model and preprocessing artifacts..."):
        model, scaler, columns = load_model_and_artifacts()

    if model is None or scaler is None or columns is None:
        st.error("Failed to load model artifacts. Please check if the required files exist.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.write("**Model Type:** Gradient Boosting Classifier")
        st.write(f"**Features:** {len(columns)}")
        st.write("**Training Size:** ~9,000 transactions")
        st.write("**Test Accuracy:** ~99.6%")

        st.header("How It Works")
        st.write("1. Enter transaction details")
        st.write("2. System analyzes features")
        st.write("3. Model predicts fraud probability")
        st.write("4. Get instant results")

        st.header("Important Notice")
        st.warning("""
        This is a demonstration model. Always verify transactions manually
        before making decisions. False positives/negatives may occur.
        """)

    # Main interface
    input_values, metadata = create_feature_input_form(columns)

    st.markdown("---")

    # Prediction button
    if st.button("🔍 Analyze Transaction", use_container_width=True):
        is_valid, validation_message = validate_inputs(input_values, columns)

        if not is_valid:
            st.error(f"❌ {validation_message}")
        else:
            with st.spinner("Analyzing transaction..."):
                features = prepare_features(input_values, scaler, columns)

                if features is not None:
                    prediction, confidence = predict_fraud(model, features)

                    if prediction is not None:
                        display_results(prediction, confidence, metadata)
                    else:
                        st.error("Failed to make prediction. Please check the input values.")
                else:
                    st.error("Failed to prepare features for prediction.")


if __name__ == "__main__":
    main()
