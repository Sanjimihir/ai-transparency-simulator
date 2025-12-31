# AI Transparency Simulator
### Loan Decision Explainability & Trust Simulation

## Overview

The AI Transparency Simulator is an interactive Streamlit-based application that demonstrates
AI transparency, explainability, and user trust in automated loan approval systems.

The project simulates how different explanation techniques—ranging from no explanation
to SHAP-based explanations and counterfactual reasoning—affect user trust in
AI-driven financial decisions.

This simulator is designed for finance analytics, explainable AI (XAI) education,
AI ethics research, and transparency-focused system design.

---

## Objectives

- Simulate automated loan approval decisions
- Provide multiple levels of model explainability
- Enable interactive user input and scenario testing
- Capture user trust under different explanation modes
- Demonstrate responsible AI practices in financial decision systems

---

## Explanation Modes

1. No Explanation (Control)
   - Displays only the prediction and probability

2. Basic Explanation
   - Shows top contributing features using model feature importance

3. Detailed Explanation (SHAP)
   - Displays SHAP values to explain individual predictions

4. Counterfactual Explanations
   - Suggests minimal feature changes required to alter the decision outcome

---

## System Workflow

User Input  
↓  
Streamlit Interface  
↓  
ML Pipeline (Random Forest)  
↓  
Prediction (Approve / Reject)  
↓  
Explanation Layer  
- Feature Importance  
- SHAP Values  
- Counterfactual Generator  
↓  
User Trust Feedback  

---

## Dataset

- German Credit Dataset
- Binary classification task:
  - 1 → Creditworthy
  - 0 → Non-creditworthy
- Widely used in credit-risk modeling and fairness research

---

## Technology Stack

- Language: Python
- Frontend: Streamlit
- Model: Random Forest (scikit-learn)
- Explainability: SHAP
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib
- Model Persistence: Joblib

---

## Project Structure

ai-transparency-simulator/

├── app.py  
├── model.joblib  
├── requirements.txt  
├── README.md  

├── data/  
│   └── german_credit.csv  

├── src/  
│   ├── data_utils.py  
│   ├── model.py  
│   ├── explainer.py  
│   └── counterfactuals.py  

---

## How to Run Locally

Step 1: Clone the Repository

git clone https://github.com/YOUR_USERNAME/ai-transparency-simulator.git  
cd ai-transparency-simulator  

Step 2: Install Dependencies

pip install -r requirements.txt  

Step 3: Launch the Application

streamlit run app.py  

The application will open automatically in your browser.

---

## Model Evaluation

- Evaluation Metric: ROC-AUC
- Test performance is computed and displayed within the application
- Ensures transparency in both prediction and performance

---

## Key Features

- Interactive applicant selection
- Manual feature input support
- Real-time prediction probabilities
- Visual SHAP explanations
- Counterfactual decision suggestions
- Trust score collection and storage
- Robust handling of edge cases and small datasets

---

## Ethical & Research Significance

This project supports Responsible AI principles by emphasizing:
- Transparency
- Interpretability
- User trust
- Human-centered decision-making

It can be used as:
- A teaching tool for explainable AI
- A prototype for ethical financial AI systems
- A foundation for academic or applied research

---

## Future Enhancements

- Fairness and bias metrics
- Model comparison (Logistic Regression, XGBoost)
- Advanced counterfactual optimization
- User study analytics dashboard
- Integration with real-world financial datasets


This project is intended for educational and research purposes.
You are free to fork, modify, and extend it with appropriate attribution.
