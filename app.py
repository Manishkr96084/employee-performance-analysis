"""
Employee Performance & Satisfaction Analysis - Interactive Web Application
Built with Streamlit and Scikit-Learn
Author: Swayam / Final Year Project Team
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Employee Performance & Satisfaction Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid #edf2f7;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #718096;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
        margin-top: 0.3rem;
    }
    .risk-high {
        background-color: #fff5f5;
        border-left: 5px solid #e53e3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #f0fff4;
        border-left: 5px solid #38a169;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'employee_data.xlsx')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'employee_data_cleaned.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'predictions.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'src', 'models', 'rf_model.pkl')

MAPPINGS = {
    'Gender': {'Female': 0, 'Male': 1},
    'EducationBackground': {'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2, 'Medical': 3, 'Other': 4, 'Technical Degree': 5},
    'MaritalStatus': {'Divorced': 0, 'Married': 1, 'Single': 2},
    'EmpDepartment': {'Finance': 0, 'Human Resources': 1, 'Research & Development': 2, 'Sales': 3},
    'EmpJobRole': {
        'Finance Manager': 0, 'Healthcare Representative': 1, 'Human Resources': 2,
        'Laboratory Technician': 3, 'Manager': 4, 'Manager R&D': 5,
        'Manufacturing Director': 6, 'Research Director': 7, 'Research Scientist': 8,
        'Sales Executive': 9, 'Sales Representative': 10, 'Senior Manager R&D': 11
    },
    'BusinessTravelFrequency': {'Non-Travel': 0, 'Travel_Frequently': 1, 'Travel_Rarely': 2},
    'OverTime': {'No': 0, 'Yes': 1},
    'Attrition': {'No': 0, 'Yes': 1}
}

INVERSE_MAPPINGS = {col: {v: k for k, v in m.items()} for col, m in MAPPINGS.items()}

@st.cache_data
def load_datasets():
    raw_df = pd.read_excel(RAW_DATA_PATH) if os.path.exists(RAW_DATA_PATH) else None
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH) if os.path.exists(CLEANED_DATA_PATH) else None
    preds_df = pd.read_csv(PREDICTIONS_PATH) if os.path.exists(PREDICTIONS_PATH) else None
    return raw_df, cleaned_df, preds_df

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

raw_df, cleaned_df, preds_df = load_datasets()
model = load_model()

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1522071820081-009f0129c71c?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.markdown("### 🏢 INX Future Inc.")
    st.markdown("**Employee Satisfaction & Performance Analytics**")
    st.caption("IABAC Certified Data Scientist Reference Project")
    st.divider()

    navigation = st.radio(
        "Navigate Application",
        [
            "📊 Executive Dashboard",
            "🔍 Exploratory Data Analysis",
            "🧠 Model Performance & Drivers",
            "🎯 Live Satisfaction Predictor",
            "📋 Batch Employee Monitor",
            "📖 Project Documentation"
        ]
    )

    st.divider()
    st.markdown("#### ⚡ System Status")
    st.write(f"• Raw Dataset: {'✅ Loaded' if raw_df is not None else '❌ Missing'}")
    st.write(f"• Cleaned Data: {'✅ Ready' if cleaned_df is not None else '❌ Missing'}")
    st.write(f"• RF Model: {'✅ Loaded' if model is not None else '❌ Not Found'}")
    st.write(f"• Predictions: {'✅ Generated' if preds_df is not None else '❌ Missing'}")

# Header
st.markdown('<div class="main-header">Employee Performance & Satisfaction Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-driven workforce diagnostics, predictive risk modeling, and HR intervention strategy</div>', unsafe_allow_html=True)

# ----------------- TAB 1: EXECUTIVE DASHBOARD -----------------
if navigation == "📊 Executive Dashboard":
    st.subheader("Workforce Diagnostic Overview")

    if raw_df is not None and preds_df is not None:
        total_emp = len(raw_df)
        low_sat_count = (raw_df['EmpJobSatisfaction'] <= 2).sum()
        low_sat_pct = (low_sat_count / total_emp) * 100
        attrition_count = (raw_df['Attrition'] == 'Yes').sum()
        attrition_pct = (attrition_count / total_emp) * 100
        high_risk_count = (preds_df['HighRiskFlag'] == 1).sum() if 'HighRiskFlag' in preds_df.columns else 0
        high_risk_pct = (high_risk_count / total_emp) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Employees</div>
                <div class="metric-value">{total_emp}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Low Satisfaction Rate</div>
                <div class="metric-value" style="color: #dd6b20;">{low_sat_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted High Risk</div>
                <div class="metric-value" style="color: #e53e3e;">{high_risk_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Model Accuracy</div>
                <div class="metric-value" style="color: #38a169;">72.2%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("##### 🏢 Department-wise Satisfaction Breakdown")
            dept_summary = raw_df.groupby('EmpDepartment')['EmpJobSatisfaction'].agg(
                Average_Satisfaction='mean',
                Total_Count='count',
                Low_Satisfaction_Count=lambda x: (x <= 2).sum()
            ).reset_index()
            dept_summary['Low_Satisfaction_Pct'] = (dept_summary['Low_Satisfaction_Count'] / dept_summary['Total_Count']) * 100

            fig_dept = px.bar(
                dept_summary,
                x='EmpDepartment',
                y='Low_Satisfaction_Pct',
                color='EmpDepartment',
                text=dept_summary['Low_Satisfaction_Pct'].apply(lambda x: f"{x:.1f}%"),
                labels={'Low_Satisfaction_Pct': '% Low Satisfaction (≤ 2)', 'EmpDepartment': 'Department'},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_dept.update_traces(textposition='outside')
            fig_dept.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_dept, use_container_width=True)

        with c2:
            st.markdown("##### ⏱️ Impact of OverTime on Employee Satisfaction")
            ot_fig = px.box(
                raw_df,
                x='OverTime',
                y='EmpJobSatisfaction',
                color='OverTime',
                labels={'EmpJobSatisfaction': 'Satisfaction Level (1-4)', 'OverTime': 'Works OverTime'},
                color_discrete_sequence=['#4299e1', '#f56565']
            )
            ot_fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(ot_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 📋 Employee Dataset Inspection")
        view_type = st.radio("Select View:", ["Raw Human-Readable Data", "Processed & Encoded Data"], horizontal=True)
        if view_type == "Raw Human-Readable Data":
            st.dataframe(raw_df, use_container_width=True, height=280)
        else:
            st.dataframe(cleaned_df, use_container_width=True, height=280)

# ----------------- TAB 2: EXPLORATORY DATA ANALYSIS -----------------
elif navigation == "🔍 Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis & Feature Relationships")

    if raw_df is not None:
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Satisfaction & Work Factors", "Performance Analysis", "Correlation Matrix"])

        with eda_tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Job Satisfaction Distribution")
                fig_sat = px.histogram(
                    raw_df,
                    x='EmpJobSatisfaction',
                    color_discrete_sequence=['#3182ce'],
                    labels={'EmpJobSatisfaction': 'Job Satisfaction (1: Low, 4: High)', 'count': 'Employee Count'}
                )
                fig_sat.update_layout(bargap=0.2, height=350)
                st.plotly_chart(fig_sat, use_container_width=True)

            with col2:
                st.markdown("##### Hourly Rate vs Job Satisfaction")
                fig_hr = px.violin(
                    raw_df,
                    x='EmpJobSatisfaction',
                    y='EmpHourlyRate',
                    box=True,
                    color='EmpJobSatisfaction',
                    labels={'EmpJobSatisfaction': 'Satisfaction Level', 'EmpHourlyRate': 'Hourly Rate ($)'}
                )
                fig_hr.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_hr, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("##### Environment Satisfaction vs Job Satisfaction")
                fig_env = px.density_heatmap(
                    raw_df,
                    x='EmpEnvironmentSatisfaction',
                    y='EmpJobSatisfaction',
                    text_auto=True,
                    labels={'EmpEnvironmentSatisfaction': 'Environment Satisfaction', 'EmpJobSatisfaction': 'Job Satisfaction'},
                    color_continuous_scale='Blues'
                )
                fig_env.update_layout(height=350)
                st.plotly_chart(fig_env, use_container_width=True)

            with col4:
                st.markdown("##### Distance From Home vs Low Satisfaction")
                raw_df['LowSatisfactionStr'] = raw_df['EmpJobSatisfaction'].apply(lambda x: 'Low (≤2)' if x <= 2 else 'Normal/High (3-4)')
                fig_dist = px.box(
                    raw_df,
                    x='LowSatisfactionStr',
                    y='DistanceFromHome',
                    color='LowSatisfactionStr',
                    labels={'LowSatisfactionStr': 'Satisfaction Status', 'DistanceFromHome': 'Distance from Home (km)'},
                    color_discrete_sequence=['#e53e3e', '#319795']
                )
                fig_dist.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)

        with eda_tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Performance Rating Distribution by Department")
                fig_perf = px.histogram(
                    raw_df,
                    x='PerformanceRating',
                    color='EmpDepartment',
                    barmode='group',
                    labels={'PerformanceRating': 'Performance Rating', 'count': 'Count'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_perf.update_layout(height=380)
                st.plotly_chart(fig_perf, use_container_width=True)

            with c2:
                st.markdown("##### Salary Hike Percentage vs Performance Rating")
                fig_hike = px.box(
                    raw_df,
                    x='PerformanceRating',
                    y='EmpLastSalaryHikePercent',
                    color='PerformanceRating',
                    labels={'PerformanceRating': 'Performance Rating', 'EmpLastSalaryHikePercent': 'Salary Hike %'}
                )
                fig_hike.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_hike, use_container_width=True)

        with eda_tab3:
            st.markdown("##### Feature Correlation Heatmap")
            if cleaned_df is not None:
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.drop(['EmpNumber'], errors='ignore')
                corr = cleaned_df[numeric_cols].corr()

                fig_corr = px.imshow(
                    corr,
                    text_auto=False,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )
                fig_corr.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_corr, use_container_width=True)

# ----------------- TAB 3: MODEL PERFORMANCE & DRIVERS -----------------
elif navigation == "🧠 Model Performance & Drivers":
    st.subheader("Random Forest Classifier Diagnostics")

    if model is not None and cleaned_df is not None:
        df_model = cleaned_df.copy()
        df_model['LowSatisfaction'] = (df_model['EmpJobSatisfaction'] <= 2).astype(int)
        target = 'LowSatisfaction'
        X = df_model.drop(columns=['EmpNumber', 'EmpJobSatisfaction', target, 'PredictedLowSatisfaction', 'RiskProbability', 'HighRiskFlag'], errors='ignore')
        y = df_model[target]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### 🎯 Model Metrics")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            m1, m2 = st.columns(2)
            m1.metric("Validation Accuracy", f"{acc * 100:.1f}%", "+22% over baseline")
            m2.metric("Target Split", f"{y.sum()} Low / {len(y)-y.sum()} Normal")

            st.markdown("##### Confusion Matrix")
            cm_df = pd.DataFrame(cm, index=['Actual Normal (0)', 'Actual Low (1)'], columns=['Pred Normal (0)', 'Pred Low (1)'])
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
            fig_cm.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.markdown("##### 🏆 Top Drivers of Low Satisfaction (Feature Importance)")
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)

            fig_imp = px.bar(
                feat_df.tail(12),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='teal',
                text=feat_df.tail(12)['Importance'].apply(lambda x: f"{x:.3f}")
            )
            fig_imp.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Detailed Classification Report")
        rep = classification_report(y_test, y_pred, output_dict=True)
        rep_df = pd.DataFrame(rep).transpose()
        st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

# ----------------- TAB 4: LIVE PREDICTOR -----------------
elif navigation == "🎯 Live Satisfaction Predictor":
    st.subheader("Predict Employee Low Satisfaction Risk")
    st.markdown("Enter employee attributes to simulate satisfaction risk and generate real-time HR recommendations.")

    if model is not None and cleaned_df is not None:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 👤 Demographics & Department")
                age = st.slider("Age", 18, 65, 34)
                gender = st.selectbox("Gender", list(MAPPINGS['Gender'].keys()))
                marital = st.selectbox("Marital Status", list(MAPPINGS['MaritalStatus'].keys()))
                dept = st.selectbox("Department", list(MAPPINGS['EmpDepartment'].keys()))
                role = st.selectbox("Job Role", list(MAPPINGS['EmpJobRole'].keys()))
                edu_bg = st.selectbox("Education Background", list(MAPPINGS['EducationBackground'].keys()))
                edu_lvl = st.slider("Education Level (1-5)", 1, 5, 3)

            with col2:
                st.markdown("##### 💼 Role & Work Conditions")
                overtime = st.selectbox("OverTime", ["No", "Yes"], index=1)
                travel = st.selectbox("Business Travel", list(MAPPINGS['BusinessTravelFrequency'].keys()))
                dist_home = st.slider("Distance From Home (km)", 1, 30, 10)
                hourly_rate = st.slider("Hourly Rate ($)", 30, 100, 65)
                job_lvl = st.slider("Job Level (1-5)", 1, 5, 2)
                env_sat = st.slider("Environment Satisfaction (1: Low - 4: High)", 1, 4, 2)
                job_inv = st.slider("Job Involvement (1: Low - 4: High)", 1, 4, 2)

            with col3:
                st.markdown("##### 📈 Career & Experience")
                total_exp = st.slider("Total Work Experience (Years)", 0, 40, 8)
                exp_curr_company = st.slider("Years at This Company", 0, 30, 4)
                exp_curr_role = st.slider("Years in Current Role", 0, 20, 3)
                years_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
                years_curr_mgr = st.slider("Years With Current Manager", 0, 20, 2)
                work_life = st.slider("Work Life Balance (1: Bad - 4: Best)", 1, 4, 2)
                rel_sat = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
                num_comp = st.slider("Num Companies Worked", 0, 10, 2)
                salary_hike = st.slider("Last Salary Hike (%)", 10, 25, 14)
                train_times = st.slider("Training Times Last Year", 0, 6, 2)
                attrition = st.selectbox("Attrition Status", ["No", "Yes"], index=0)
                perf_rating = st.selectbox("Performance Rating", [2, 3, 4], index=1)

            submit_btn = st.form_submit_button("🔮 Predict Employee Satisfaction Risk", use_container_width=True)

        if submit_btn:
            input_dict = {
                'Age': age,
                'Gender': MAPPINGS['Gender'][gender],
                'EducationBackground': MAPPINGS['EducationBackground'][edu_bg],
                'MaritalStatus': MAPPINGS['MaritalStatus'][marital],
                'EmpDepartment': MAPPINGS['EmpDepartment'][dept],
                'EmpJobRole': MAPPINGS['EmpJobRole'][role],
                'BusinessTravelFrequency': MAPPINGS['BusinessTravelFrequency'][travel],
                'DistanceFromHome': dist_home,
                'EmpEducationLevel': edu_lvl,
                'EmpEnvironmentSatisfaction': env_sat,
                'EmpHourlyRate': hourly_rate,
                'EmpJobInvolvement': job_inv,
                'EmpJobLevel': job_lvl,
                'NumCompaniesWorked': num_comp,
                'OverTime': MAPPINGS['OverTime'][overtime],
                'EmpLastSalaryHikePercent': salary_hike,
                'EmpRelationshipSatisfaction': rel_sat,
                'TotalWorkExperienceInYears': total_exp,
                'TrainingTimesLastYear': train_times,
                'EmpWorkLifeBalance': work_life,
                'ExperienceYearsAtThisCompany': exp_curr_company,
                'ExperienceYearsInCurrentRole': exp_curr_role,
                'YearsSinceLastPromotion': years_promotion,
                'YearsWithCurrManager': years_curr_mgr,
                'Attrition': MAPPINGS['Attrition'][attrition],
                'PerformanceRating': perf_rating
            }

            input_df = pd.DataFrame([input_dict])
            expected_cols = list(model.feature_names_in_)
            input_df = input_df[expected_cols]

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.markdown("### 📊 Assessment Outcome")
            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                if prob >= 0.60:
                    st.markdown(f"""
                    <div class="risk-high">
                        <h3 style="color: #c53030; margin-top:0;">⚠️ High Risk of Low Job Satisfaction</h3>
                        <p><strong>Satisfaction Risk Probability: {prob * 100:.1f}%</strong></p>
                        <p>This employee exhibits key behavioural and environmental risk indicators strongly associated with dissatisfaction and potential attrition.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <h3 style="color: #276749; margin-top:0;">✅ Healthy Engagement Profile</h3>
                        <p><strong>Satisfaction Risk Probability: {prob * 100:.1f}%</strong></p>
                        <p>The employee shows positive engagement drivers consistent with stable retention and satisfaction.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with res_c2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "Risk Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#e53e3e" if prob >= 0.60 else "#38a169"},
                        'steps': [
                            {'range': [0, 40], 'color': "#edf2f7"},
                            {'range': [40, 60], 'color': "#feebc8"},
                            {'range': [60, 100], 'color': "#fed7d7"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))
                fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Recommendations
            st.markdown("##### 💡 Targeted HR Interventions")
            recs = []
            if overtime == "Yes":
                recs.append("⏱️ **Manage Workload:** OverTime is one of the highest drivers of burnout. Implement workload rebalancing or compensatory off.")
            if env_sat <= 2:
                recs.append("🏢 **Work Environment Audit:** Low environment satisfaction detected. Arrange an informal sync to address tooling, peer dynamics, or physical workspace comfort.")
            if work_life <= 2:
                recs.append("⚖️ **Flexibility / WFH Policy:** Offer flexible working hours or remote days to ease work-life friction.")
            if years_promotion >= 3:
                recs.append("🚀 **Career Progression:** Review promotion and recognition cycle; stagnancy in role contributes to dissatisfaction.")
            if not recs:
                recs.append("🌟 **Sustain Momentum:** The current engagement parameters are optimal. Maintain periodic check-ins and recognition.")

            for r in recs:
                st.info(r)

# ----------------- TAB 5: BATCH EMPLOYEE MONITOR -----------------
elif navigation == "📋 Batch Employee Monitor":
    st.subheader("Workforce Risk Surveillance & Batch Records")

    if preds_df is not None and raw_df is not None:
        merged_df = raw_df.copy()
        if 'PredictedLowSatisfaction' in preds_df.columns:
            merged_df['Predicted_Low_Sat'] = preds_df['PredictedLowSatisfaction']
        if 'RiskProbability' in preds_df.columns:
            merged_df['Risk_Probability'] = (preds_df['RiskProbability'] * 100).round(1)
        if 'HighRiskFlag' in preds_df.columns:
            merged_df['High_Risk_Flag'] = preds_df['HighRiskFlag'].apply(lambda x: '🚨 High Risk' if x == 1 else '✅ Normal')

        col1, col2, col3 = st.columns(3)
        with col1:
            dept_filter = st.multiselect("Filter by Department:", options=merged_df['EmpDepartment'].unique(), default=merged_df['EmpDepartment'].unique())
        with col2:
            risk_filter = st.multiselect("Filter by Risk Level:", options=['🚨 High Risk', '✅ Normal'], default=['🚨 High Risk', '✅ Normal'])
        with col3:
            overtime_filter = st.multiselect("Filter by OverTime:", options=merged_df['OverTime'].unique(), default=merged_df['OverTime'].unique())

        filtered = merged_df[
            (merged_df['EmpDepartment'].isin(dept_filter)) &
            (merged_df['High_Risk_Flag'].isin(risk_filter)) &
            (merged_df['OverTime'].isin(overtime_filter))
        ]

        st.markdown(f"**Showing {len(filtered)} out of {len(merged_df)} employee records**")
        st.dataframe(filtered, use_container_width=True, height=350)

        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Records as CSV",
            data=csv,
            file_name="employee_risk_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

# ----------------- TAB 6: PROJECT DOCUMENTATION -----------------
elif navigation == "📖 Project Documentation":
    st.subheader("Project Architecture & IABAC Certification Reference")

    st.markdown("""
    ### 🎯 Project Overview
    This project addresses workforce engagement and retention challenges at **INX Future Inc.** by building an end-to-end data analytics and machine learning solution.

    ### 🧱 System Workflow
    1. **Data Preprocessing (`data_processing.ipynb` / `main.py`)**:
       - Cleans missing values, handles data types, encodes 8 categorical attributes.
       - Generates `employee_data_cleaned.csv`.
    2. **Exploratory Data Analysis (`data_exploratory_analysis.ipynb`)**:
       - Analyzes department-wise satisfaction variations, overtime impacts, and correlation structures.
    3. **Predictive Modeling (`train_model.ipynb`)**:
       - Trains an ensemble **Random Forest Classifier** to detect employees at risk of low satisfaction (`JobSatisfaction ≤ 2`).
       - Achieves **72.2% test accuracy**.
       - Evaluates feature importances highlighting **TotalWorkExperience**, **HourlyRate**, **Age**, and **OverTime**.
    4. **Prediction Pipeline (`predict_model.ipynb`)**:
       - Scores workforce records and assigns high-risk probability flags (>60%).
    5. **Interactive Dashboard (`app.py`)**:
       - Live executive reporting and individual employee scenario simulation.
    """)
