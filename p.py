import joblib
import streamlit as st

# Load the trained model and scaler
classifier = joblib.load('classifier.pkl')
scaler = joblib.load('scaler.pkl')

@st.cache_data()
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
    # Pre-process user input
    Gender = 0 if Gender == "Male" else 1
    Married = 0 if Married == "Unmarried" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1
    LoanAmount = LoanAmount / 1000  # Scale loan amount if required

    # Scale the input features
    features = scaler.transform([[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])

    # Predict and interpret result
    prediction = classifier.predict(features)
    return 'Approved' if prediction == 1 else 'Rejected'

# Streamlit Interface
def main():
    # Set up page config with custom theme colors
    st.set_page_config(page_title="Smart Loan Approval Advisor", page_icon="💼", layout="centered")

    # Custom background color and style for the app
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f6;
            color: #4c4c6d;
        }
        .main-button {
            background-color: #3498db;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("💼 Smart Loan Approval Advisor")
    st.write("#### Make data-driven loan decisions with confidence.")

    # Input fields for user data with icons for a modern look
    st.markdown("### Application Details")
    Gender = st.radio("Select your Gender:", ("Male", "Female"), help="Choose your gender.")
    Married = st.radio("Marital Status:", ("Unmarried", "Married"), help="Choose your marital status.")
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, help="Enter your monthly income.")
    LoanAmount = st.slider("Loan Amount Requested (in USD)", min_value=0, max_value=500000, step=1000, help="Choose the loan amount you're applying for.")
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"), help="Choose your credit status.")

    # Customized button with unique color
    if st.button("Predict My Loan Status", key="predict_button"):
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
        st.success(f'Your loan application status: **{result}**', icon="✅" if result == "Approved" else "❌")

    # Additional Info Section with FAQ-style dropdown
    st.write("---")
    with st.expander("How does this app make predictions?"):
        st.write("""
            This app uses a machine learning model trained on historical loan data. It evaluates various factors
            such as gender, marital status, income, loan amount, and credit history to predict if a loan
            application will be approved or rejected.
        """)
    with st.expander("Is my information secure?"):
        st.write("""
            Yes, this app runs locally on your device, and your data is not stored or shared.
        """)
    with st.expander("Why was my application rejected?"):
        st.write("""
            Loan rejections could be due to low income, high requested loan amount, or unclear credit history.
            Each factor affects your application, and approval is not guaranteed.
        """)

if __name__ == '__main__':
    main()
