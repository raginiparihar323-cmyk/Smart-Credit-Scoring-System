import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

model = pickle.load(open("credit_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_lottie_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

finance_animation = load_lottie_url(
    "https://assets10.lottiefiles.com/packages/lf20_tno6cg2w.json"
)

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at top left, #1e3a8a, #020617 45%, #020617);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}

.hero {
    padding: 40px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(37,99,235,0.35), rgba(14,165,233,0.12));
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0px 20px 45px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 52px;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
}

.glass-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    padding: 24px;
    border-radius: 24px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
    backdrop-filter: blur(14px);
    margin-bottom: 20px;
}

.result-good {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    padding: 28px;
    border-radius: 24px;
    color: white;
    font-size: 28px;
    font-weight: 900;
    text-align: center;
    box-shadow: 0 15px 35px rgba(34,197,94,0.28);
}

.result-bad {
    background: linear-gradient(135deg, #dc2626, #f97316);
    padding: 28px;
    border-radius: 24px;
    color: white;
    font-size: 28px;
    font-weight: 900;
    text-align: center;
    box-shadow: 0 15px 35px rgba(249,115,22,0.28);
}

.small-note {
    color: #cbd5e1;
    font-size: 15px;
}

.metric-box {
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.14);
}

.metric-title {
    color: #cbd5e1;
    font-size: 14px;
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: white;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 14px;
    color: white;
    padding: 12px 20px;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 14px;
    font-size: 18px;
    font-weight: 800;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
    color: white;
}

label, .stMarkdown, .stText {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("💳 Credit Risk AI")
st.sidebar.markdown("### Project Dashboard")
st.sidebar.write("This app predicts customer credit risk using Machine Learning.")

st.sidebar.markdown("---")
st.sidebar.write("### Target Meaning")
st.sidebar.success("0 = Good Customer")
st.sidebar.error("1 = Risky / Default Customer")

st.sidebar.markdown("---")
st.sidebar.write("### PAY Status")
st.sidebar.write("""
- -2 = No usage  
- -1 = Paid fully  
- 0 = Normal payment  
- 1 = Delay 1 month  
- 2+ = Delay 2 or more months  
""")

st.sidebar.markdown("---")
st.sidebar.info("Model: Random Forest Classifier")

hero_left, hero_right = st.columns([1.8, 1])

with hero_left:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">AI Credit Scoring Dashboard</div>
        <div class="hero-subtitle">
            Predict customer default risk using financial behavior, repayment history,
            billing pattern and machine learning intelligence.
        </div>
        <br>
        <span class="small-note">Built for Credit Scoring System using Machine Learning</span>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
   if finance_animation:
    st_lottie(finance_animation, height=260)
   else:
       st.markdown("""
       <div class="glass-card" style="text-align:center; padding:45px;">
           <h1 style="font-size:70px;">💳</h1>
           <h2>Smart Risk Engine</h2>
           <p class="small-note">AI-powered credit default prediction</p>
      </div>
      """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "👤 Customer Profile",
    "💰 Financial Behaviour",
    "📊 Risk Intelligence"
])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("👤 Personal Details")

        LIMIT_BAL = st.number_input("💳 Credit Limit", min_value=0, value=50000, step=10000)
        AGE = st.number_input("🎂 Age", min_value=18, max_value=100, value=25, step=1)

        SEX = st.selectbox(
            "⚧ Sex",
            [1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female"
        )

        EDUCATION = st.selectbox(
            "🎓 Education",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Graduate School",
                2: "University",
                3: "High School",
                4: "Others"
            }[x]
        )

        MARRIAGE = st.selectbox(
            "💍 Marital Status",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Married",
                2: "Single",
                3: "Others"
            }[x]
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📅 Repayment History")

        pay_options = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

        PAY_0 = st.selectbox("Latest Month Status", pay_options, index=2)
        PAY_2 = st.selectbox("2 Months Ago Status", pay_options, index=2)
        PAY_3 = st.selectbox("3 Months Ago Status", pay_options, index=2)
        PAY_4 = st.selectbox("4 Months Ago Status", pay_options, index=2)
        PAY_5 = st.selectbox("5 Months Ago Status", pay_options, index=2)
        PAY_6 = st.selectbox("6 Months Ago Status", pay_options, index=2)

        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💰 Monthly Billing and Payment Pattern")

    bill_col, pay_col = st.columns(2)

    with bill_col:
        st.write("### 🧾 Bill Amounts")
        BILL_AMT = []
        for i in range(1, 7):
            BILL_AMT.append(
                st.number_input(f"Bill Amount Month {i}", value=10000, step=1000)
            )

    with pay_col:
        st.write("### 💵 Payment Amounts")
        PAY_AMT = []
        for i in range(1, 7):
            PAY_AMT.append(
                st.number_input(f"Payment Amount Month {i}", min_value=0, value=2000, step=500)
            )

    st.markdown('</div>', unsafe_allow_html=True)

TOTAL_BILL_AMT = sum(BILL_AMT)
TOTAL_PAY_AMT = sum(PAY_AMT)
PAYMENT_RATIO = TOTAL_PAY_AMT / (TOTAL_BILL_AMT + 1)

features = np.array([[
    LIMIT_BAL,
    SEX,
    EDUCATION,
    MARRIAGE,
    AGE,
    PAY_0,
    PAY_2,
    PAY_3,
    PAY_4,
    PAY_5,
    PAY_6,
    *BILL_AMT,
    *PAY_AMT,
    TOTAL_BILL_AMT,
    TOTAL_PAY_AMT,
    PAYMENT_RATIO
]])

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📌 Financial Summary")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Credit Limit</div>
            <div class="metric-value">{LIMIT_BAL:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Bill</div>
            <div class="metric-value">{TOTAL_BILL_AMT:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Payment</div>
            <div class="metric-value">{TOTAL_PAY_AMT:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Payment Ratio</div>
            <div class="metric-value">{PAYMENT_RATIO:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Analyze Credit Risk", use_container_width=True):

    with st.spinner("Analyzing customer financial behaviour..."):

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        risk_percent = probability * 100

        result_col, gauge_col = st.columns([1, 1])

    with result_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Final Credit Decision")

        if prediction == 0:
            st.markdown(
                '<div class="result-good">✅ GOOD CREDIT RISK</div>',
                unsafe_allow_html=True
            )

            st.balloons()

        else:
            st.markdown(
                '<div class="result-bad">⚠️ BAD / RISKY CREDIT RISK</div>',
                unsafe_allow_html=True
            )

            st.write("")
            st.write(f"### Default Risk Probability: **{risk_percent:.2f}%**")

            if risk_percent < 30:
                st.success("Risk Level: Low")
            elif risk_percent < 60:
                st.warning("Risk Level: Medium")
            else:
                st.error("Risk Level: High")

            st.markdown("""
            <div class="small-note">
                Business meaning: This result can support banks in identifying risky customers
                before credit approval.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with gauge_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📈 Risk Gauge")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                title={"text": "Default Risk %", "font": {"color": "white"}},
                number={"font": {"color": "white", "size": 44}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#38bdf8"},
                    "bgcolor": "rgba(255,255,255,0.08)",
                    "borderwidth": 2,
                    "bordercolor": "rgba(255,255,255,0.2)",
                    "steps": [
                        {"range": [0, 30], "color": "#166534"},
                        {"range": [30, 60], "color": "#ca8a04"},
                        {"range": [60, 100], "color": "#991b1b"}
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 4},
                        "thickness": 0.75,
                        "value": risk_percent
                    }
                }
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                height=360,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Model Performance")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown('<div class="metric-box"><div class="metric-title">Accuracy</div><div class="metric-value">78.8%</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="metric-box"><div class="metric-title">Recall</div><div class="metric-value">55.3%</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown('<div class="metric-box"><div class="metric-title">F1 Score</div><div class="metric-value">53.6%</div></div>', unsafe_allow_html=True)
    with p4:
        st.markdown('<div class="metric-box"><div class="metric-title">ROC-AUC</div><div class="metric-value">77.3%</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
