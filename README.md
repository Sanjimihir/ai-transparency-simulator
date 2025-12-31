AI Transparency Simulator
Loan Decision Explainability & Trust Simulation
📌 Overview

The AI Transparency Simulator is an interactive Streamlit-based application designed to demonstrate AI explainability, transparency, and user trust in automated loan approval systems.

The project simulates how different explanation mechanisms—ranging from no explanation to SHAP-based explanations and counterfactual reasoning—affect user trust in AI-driven financial decisions.

This tool is intended for AI ethics research, finance analytics, explainable AI (XAI) education, and transparency studies.

🎯 Objectives

Simulate automated loan approval decisions

Provide multiple levels of model explainability

Allow users to interact with applicant data

Measure user trust under different explanation modes

Demonstrate responsible AI practices in the finance domain

🧠 Explanation Modes Supported

No Explanation (Control Group)

Only shows the model decision and probability

Basic Explanation

Displays top contributing features using model feature importance

Detailed Explanation (SHAP)

Uses SHAP values to show feature-level contribution to predictions

Counterfactual Explanations

Suggests minimal feature changes required to flip the decision outcome

🏗️ System Architecture
User Input
   ↓
Streamlit Interface
   ↓
ML Pipeline (Random Forest)
   ↓
Prediction (Approve / Reject)
   ↓
Explanation Layer
   ├── Feature Importance
   ├── SHAP Values
   └── Counterfactual Generator
   ↓
Trust Feedback Collection

🧪 Dataset

German Credit Dataset

Used for binary classification:

1 → Creditworthy

0 → Non-creditworthy

Commonly used in credit-risk and fairness research

🛠️ Tech Stack
Category	Tools
Language	Python
Frontend	Streamlit
ML Model	Random Forest
Explainability	SHAP
Data Handling	Pandas, NumPy
Visualization	Matplotlib
Model Storage	Joblib
📂 Project Structure
ai-transparency-simulator/
│
├── app.py                  # Main Streamlit application
├── model.joblib             # Trained ML pipeline
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
│
├── data/
│   └── german_credit.csv    # Dataset
│
├── src/
│   ├── data_utils.py        # Data loading & preprocessing
│   ├── model.py             # Model training & persistence
│   ├── explainer.py         # SHAP & explanation wrapper
│   └── counterfactuals.py   # Counterfactual generation logic

🚀 How to Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/ai-transparency-simulator.git
cd ai-transparency-simulator

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py


The app will open automatically in your browser.

📊 Model Evaluation

Metric Used: ROC-AUC

Displayed dynamically inside the application

Helps validate predictive performance alongside explainability

🔍 Key Features

Interactive applicant selection

Manual feature input option

Real-time prediction probabilities

Visual SHAP plots

Counterfactual recommendations

Trust score capture and storage

Robust error handling for small datasets

⚖️ Ethical & Research Relevance

This project aligns with modern Responsible AI principles:

Transparency

Interpretability

User trust

Human-in-the-loop decision-making

It can be used as:

A teaching tool for XAI concepts

A prototype for ethical AI systems in finance

A foundation for academic or applied research

📈 Possible Extensions

Fairness metrics (bias detection across demographics)

Model comparison (Logistic Regression vs RF)

Advanced counterfactual optimization

User study analytics dashboard

Integration with real-world financial datasets
