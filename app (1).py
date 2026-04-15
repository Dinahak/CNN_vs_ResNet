import streamlit as st
import numpy as np
import torch
import torch.nn as nn

st.set_page_config(page_title="ChurnGuard | Bank Churn Predictor", page_icon="🏦", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; }
.main { background-color: #f0f4f8; }
.header-banner { background: linear-gradient(135deg, #0a2342 0%, #1a3f6f 60%, #1e5fa8 100%); padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem; color: white; }
.header-banner h1 { font-size: 2rem; font-weight: 700; margin: 0; color: white; }
.header-banner p  { font-size: 0.95rem; margin: 0.4rem 0 0; color: #a8c4e0; }
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card { background: white; border-radius: 10px; padding: 1rem 1.25rem; flex: 1; border-left: 4px solid #1e5fa8; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
.metric-card .label { font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0a2342; margin-top: 2px; }
.metric-card .sub   { font-size: 11px; color: #9ca3af; margin-top: 2px; }
.section-title { font-size: 0.8rem; font-weight: 700; color: #1e5fa8; text-transform: uppercase; letter-spacing: 0.08em; margin: 1.2rem 0 0.6rem; }
.result-churn { background: #fff1f2; border: 2px solid #f87171; border-radius: 12px; padding: 1.5rem; text-align: center; }
.result-safe  { background: #f0fdf4; border: 2px solid #4ade80; border-radius: 12px; padding: 1.5rem; text-align: center; }
.result-label { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
.result-sub   { font-size: 0.85rem; color: #6b7280; }
.badge-improvement { display: inline-block; background: #dbeafe; color: #1e40af; border-radius: 20px; padding: 4px 14px; font-size: 12px; font-weight: 600; margin-bottom: 1rem; }
.risk-pill { display: inline-block; background: #fef3c7; color: #92400e; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 500; margin: 3px; }
.safe-pill { display: inline-block; background: #d1fae5; color: #065f46; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 500; margin: 3px; }
.compare-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #f3f4f6; font-size: 13px; }
.compare-row .metric-name { color: #6b7280; }
.compare-row .cnn-val  { color: #0a2342; font-weight: 600; }
.compare-row .res-val  { color: #1e5fa8; font-weight: 700; }
.compare-row .winner   { color: #16a34a; font-size: 11px; font-weight: 600; }
.footer { text-align: center; color: #9ca3af; font-size: 11px; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)


class DeepCNN(nn.Module):
    def __init__(self, input_size=19):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNetChurn(nn.Module):
    def __init__(self, input_size=19):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.layer1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.layer2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1), ResidualBlock(128), ResidualBlock(128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        return self.classifier(self.pool(self.layer2(self.layer1(self.stem(x))))).squeeze(1)


@st.cache_resource
def load_models():
    cnn = DeepCNN()
    res = ResNetChurn()
    try:
        cnn.load_state_dict(torch.load("cnn_model.pt", map_location="cpu"))
        res.load_state_dict(torch.load("resnet_model.pt", map_location="cpu"))
        cnn.eval(); res.eval()
        return cnn, res, True
    except FileNotFoundError:
        cnn.eval(); res.eval()
        return cnn, res, False

cnn_model, resnet_model, models_loaded = load_models()

SCALER_MEAN  = [46.32596030413745, 0.4709193245778612, 2.3462032191172115, 3.096573516342451, 1.4634146341463414, 2.8639281129653402, 0.17981633257628124, 35.928409203120374, 3.8125802310654686, 2.3411671768539546, 2.4553174681544387, 8631.953700479497, 1162.8140614199665, 7469.139638001796, 0.7599406538251061, 4404.086303939963, 64.85869457884863, 0.7122223761009566, 0.2748935518033951]
SCALER_SCALE = [8.01641820891176, 0.4991535979205121, 1.2988442163662193, 1.834721863198188, 0.7377715199413283, 1.5046254656917517, 0.6930051891558661, 7.986022008096456, 1.5543311177215595, 1.0105725007637258, 1.1061705236740995, 9088.327897373294, 814.9470959025144, 9090.236477621178, 0.21919594653094407, 3396.9615230781264, 23.471411510261287, 0.23807433569767025, 0.27567785698217995]

def predict(model, features):
    scaled = (features - np.array(SCALER_MEAN)) / np.array(SCALER_SCALE)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item()


st.markdown('<div class="header-banner"><h1>🏦 ChurnGuard</h1><p>Bank Customer Churn Prediction &nbsp;|&nbsp; ResNet Model &nbsp;|&nbsp; Week 4 — AA5750</p></div>', unsafe_allow_html=True)

if not models_loaded:
    st.info("Demo mode: model weights not found. Save them with torch.save() and rerun.", icon="ℹ️")

st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
st.markdown("""<div class="metric-row">
  <div class="metric-card"><div class="label">ResNet Accuracy</div><div class="value">95%</div><div class="sub">+1% vs Deep CNN</div></div>
  <div class="metric-card"><div class="label">Churn Recall</div><div class="value">78%</div><div class="sub">+3% vs Deep CNN</div></div>
  <div class="metric-card"><div class="label">ROC-AUC</div><div class="value">0.9796</div><div class="sub">+0.005 vs Deep CNN</div></div>
  <div class="metric-card"><div class="label">Churn Precision</div><div class="value">91%</div><div class="sub">+4% vs Deep CNN</div></div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
    st.markdown("**Account Details**")
    c1, c2 = st.columns(2)
    age                      = c1.number_input("Customer Age", 18, 100, 45)
    months_on_book           = c2.number_input("Months on Book", 0, 60, 36)
    dependent_count          = c1.number_input("Dependents", 0, 10, 2)
    total_relationship_count = c2.number_input("Products Held", 1, 6, 3)
    st.markdown("**Activity — Last 12 Months**")
    c3, c4 = st.columns(2)
    months_inactive = c3.number_input("Months Inactive", 0, 12, 2)
    contacts_count  = c4.number_input("Contacts with Bank", 0, 10, 3)
    total_trans_ct  = c3.number_input("Total Transactions", 0, 200, 60)
    total_trans_amt = c4.number_input("Total Trans. Amount ($)", 0, 20000, 4000)
    st.markdown("**Financial Profile**")
    c5, c6 = st.columns(2)
    credit_limit        = c5.number_input("Credit Limit ($)", 0, 35000, 8000)
    total_revolving_bal = c6.number_input("Revolving Balance ($)", 0, 3000, 1200)
    avg_open_to_buy     = c5.number_input("Avg Open to Buy ($)", 0, 35000, 6800)
    avg_utilization     = c6.slider("Card Utilization Ratio", 0.0, 1.0, 0.25, 0.01)
    st.markdown("**Change Ratios (Q4 vs Q1)**")
    c7, c8 = st.columns(2)
    total_amt_chng = c7.slider("Spend Change Ratio", 0.0, 4.0, 0.8, 0.01)
    total_ct_chng  = c8.slider("Transaction Count Change", 0.0, 4.0, 0.7, 0.01)
    st.markdown("**Demographics**")
    c9, c10 = st.columns(2)
    gender         = c9.selectbox("Gender", ["F", "M"])
    education      = c10.selectbox("Education Level", ["Graduate","High School","Unknown","Uneducated","College","Post-Graduate","Doctorate"])
    marital_status = c9.selectbox("Marital Status", ["Married","Single","Unknown","Divorced"])
    income_cat     = c10.selectbox("Income Category", ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Unknown"])
    card_category  = st.selectbox("Card Category", ["Blue","Silver","Gold","Platinum"])
    run = st.button("Run Churn Prediction", use_container_width=True, type="primary")

with right:
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
    if run:
        edu_map     = {"College":0,"Doctorate":1,"Graduate":2,"High School":3,"Post-Graduate":4,"Uneducated":5,"Unknown":6}
        marital_map = {"Divorced":0,"Married":1,"Single":2,"Unknown":3}
        income_map  = {"$120K +":0,"$40K - $60K":1,"$60K - $80K":2,"$80K - $120K":3,"Less than $40K":4,"Unknown":5}
        card_map    = {"Blue":0,"Gold":1,"Platinum":2,"Silver":3}
        raw = np.array([age, dependent_count, months_on_book, total_relationship_count,
                        months_inactive, contacts_count, credit_limit, total_revolving_bal,
                        avg_open_to_buy, total_amt_chng, total_trans_amt, total_trans_ct,
                        total_ct_chng, avg_utilization,
                        0 if gender == "F" else 1, edu_map[education],
                        marital_map[marital_status], income_map[income_cat],
                        card_map[card_category]], dtype=np.float32)
        cnn_prob    = predict(cnn_model, raw)
        resnet_prob = predict(resnet_model, raw)
        churn       = resnet_prob > 0.5
        if churn:
            st.markdown(f'<div class="result-churn"><div class="result-label" style="color:#dc2626;">⚠ High Churn Risk</div><div style="font-size:2.2rem;font-weight:800;color:#dc2626;margin:8px 0;">{resnet_prob*100:.1f}%</div><div class="result-sub">ResNet churn probability</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-safe"><div class="result-label" style="color:#16a34a;">✓ Low Churn Risk</div><div style="font-size:2.2rem;font-weight:800;color:#16a34a;margin:8px 0;">{resnet_prob*100:.1f}%</div><div class="result-sub">ResNet churn probability</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Churn Probability</div>', unsafe_allow_html=True)
        st.progress(resnet_prob, text=f"ResNet:   {resnet_prob*100:.1f}%")
        st.progress(cnn_prob,    text=f"Deep CNN: {cnn_prob*100:.1f}%")
        st.markdown('<div class="section-title">CNN vs ResNet Comparison</div>', unsafe_allow_html=True)
        st.markdown(f"""<div style="background:white;border-radius:10px;padding:1rem 1.25rem;box-shadow:0 1px 4px rgba(0,0,0,0.07);">
          <div class="compare-row"><span class="metric-name">Metric</span><span class="cnn-val">Deep CNN</span><span class="res-val">ResNet</span><span class="winner">Better</span></div>
          <div class="compare-row"><span class="metric-name">Test Accuracy</span><span class="cnn-val">94%</span><span class="res-val">95%</span><span class="winner">ResNet ↑</span></div>
          <div class="compare-row"><span class="metric-name">Churn Recall</span><span class="cnn-val">75%</span><span class="res-val">78%</span><span class="winner">ResNet ↑</span></div>
          <div class="compare-row"><span class="metric-name">Churn Precision</span><span class="cnn-val">87%</span><span class="res-val">91%</span><span class="winner">ResNet ↑</span></div>
          <div class="compare-row"><span class="metric-name">ROC-AUC</span><span class="cnn-val">0.9745</span><span class="res-val">0.9796</span><span class="winner">ResNet ↑</span></div>
          <div class="compare-row" style="border:none;"><span class="metric-name">This prediction</span><span class="cnn-val">{cnn_prob*100:.1f}%</span><span class="res-val">{resnet_prob*100:.1f}%</span><span class="winner">Live</span></div>
        </div><div class="badge-improvement" style="margin-top:0.75rem;">↑ ResNet is now 5% more accurate than the Week 3 baseline CNN</div>""", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Risk Factors</div>', unsafe_allow_html=True)
        risk_flags, safe_flags = [], []
        if months_inactive >= 3:          risk_flags.append("3+ months inactive")
        if contacts_count >= 4:           risk_flags.append("Frequent bank contacts")
        if total_trans_ct < 40:           risk_flags.append("Low transaction count")
        if total_ct_chng < 0.6:           risk_flags.append("Declining transaction frequency")
        if avg_utilization < 0.15:        risk_flags.append("Very low card utilization")
        if total_relationship_count <= 2: risk_flags.append("Few products held")
        if total_trans_ct >= 70:          safe_flags.append("High transaction count")
        if total_relationship_count >= 4: safe_flags.append("Multiple products")
        if avg_utilization >= 0.3:        safe_flags.append("Healthy utilization")
        if months_inactive <= 1:          safe_flags.append("Recently active")
        if risk_flags:
            st.markdown(" ".join([f'<span class="risk-pill">⚠ {f}</span>' for f in risk_flags]), unsafe_allow_html=True)
        if safe_flags:
            st.markdown(" ".join([f'<span class="safe-pill">✓ {f}</span>' for f in safe_flags]), unsafe_allow_html=True)
        if not risk_flags and not safe_flags:
            st.markdown("No strong signals detected for this customer profile.")
    else:
        st.markdown("""<div style="background:white;border-radius:12px;padding:3rem 2rem;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.07);">
          <div style="font-size:2.5rem;margin-bottom:1rem;">🏦</div>
          <div style="font-size:1rem;font-weight:600;color:#0a2342;">Fill in the customer profile and click Run Churn Prediction</div>
          <div style="font-size:0.85rem;color:#9ca3af;margin-top:0.5rem;">Results will appear here — confidence score, model comparison, and risk factors</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="footer">ChurnGuard &nbsp;|&nbsp; AA5750 Week 4 &nbsp;|&nbsp; Deep CNN (94%) vs ResNet (95%) &nbsp;|&nbsp; Built with PyTorch + Streamlit</div>', unsafe_allow_html=True)
