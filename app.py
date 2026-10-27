import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import os

# Set page config
st.set_page_config(page_title="Server Performance Predictor", layout="wide")

st.title("📊 Server Response Time Predictor")

# --- PROJECT INTRO ---
st.markdown("""
### 📖 Project Overview
Modern web services operate under continuous conditions of uncertainty. The response time a user experiences is rarely a fixed number—it fluctuates heavily based on active workloads (like CPU usage, Memory limits, and Network traffic). 

This project strictly follows the **Probability and Statistics** syllabus. Rather than guessing, we use **Multiple Linear Regression (Unit IV)** to statistically model the relationship between these workload parameters and the server's response time. By learning from historical data, the model predicts expected future delays so we can identify performance bottlenecks early.
""")

st.divider()

@st.cache_data
def load_or_generate_data():
    if os.path.exists('server_performance_data.csv'):
        return pd.read_csv('server_performance_data.csv')
    else:
        np.random.seed(42)
        time_steps = np.arange(1000)
        request_rate = np.random.randint(50, 1000, 1000)
        cpu_usage = np.clip(request_rate * 0.08 + np.random.normal(0, 10, 1000), 1, 100)
        memory_usage = np.clip(cpu_usage * 0.5 + 20 + np.random.normal(0, 5, 1000), 1, 100)
        network_load = np.clip(request_rate * 0.05 + np.random.normal(0, 2, 1000), 1, 100)
        trend = time_steps * 0.05 
        response_time = 50 + (cpu_usage * 2.5) + (memory_usage * 1.5) + (network_load * 3.0) + trend + np.random.normal(0, 20, 1000)
        df = pd.DataFrame({'Time_Step': time_steps, 'Request_Rate': request_rate, 'CPU_Usage_Pct': cpu_usage, 'Memory_Usage_Pct': memory_usage, 'Network_Load_MBps': network_load, 'Response_Time_ms': response_time})
        df.to_csv('server_performance_data.csv', index=False)
        return df

df = load_or_generate_data()

@st.cache_resource
def train_model(data):
    features = ['Request_Rate', 'CPU_Usage_Pct', 'Memory_Usage_Pct', 'Network_Load_MBps']
    X = data[features]
    y = data['Response_Time_ms']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    residuals = y_test - y_pred
    t_stat, p_value = stats.ttest_1samp(residuals, 0)
    
    return model, mae, rmse, r2, t_stat, p_value, residuals, features

model, mae, rmse, r2, t_stat, p_value, residuals, features = train_model(df)

# Sidebar for inputs
st.sidebar.header("⚙️ Live Workload Parameters")
req_rate = st.sidebar.slider("Request Rate (req/s)", 50, 1000, 500)
cpu_usage = st.sidebar.slider("CPU Usage (%)", 1, 100, 45)
mem_usage = st.sidebar.slider("Memory Usage (%)", 1, 100, 40)
net_load = st.sidebar.slider("Network Load (MB/s)", 1, 100, 25)

# Prediction
input_data = pd.DataFrame([[req_rate, cpu_usage, mem_usage, net_load]], columns=features)
prediction = model.predict(input_data)[0]

# --- EQUATION EXPLANATION ---
st.markdown("### 🧮 The Mathematics (Simply Explained)")
intercept = model.intercept_
c_req, c_cpu, c_mem, c_net = model.coef_

st.markdown("To calculate the prediction, the statistical model uses this exact regression formula:")
st.latex(rf"\text{{Response Time}} = {intercept:.1f} + ({c_req:.2f} \times \text{{ReqRate}}) + ({c_cpu:.2f} \times \text{{CPU}}) + ({c_mem:.2f} \times \text{{Memory}}) + ({c_net:.2f} \times \text{{Network}})")

st.markdown(f"""
**What this means in plain English:**
- If the server has zero active workload, the baseline delay (intercept) is **{intercept:.1f} ms**.
- For every **1% increase in CPU**, the server slows down by an additional **{c_cpu:.2f} ms**.
- For every **1 MB/s increase in Network Load**, the server slows down by **{c_net:.2f} ms**.
- The model mathematically proves which factors hurt performance the most (higher multipliers = bigger impact!).
""")

st.divider()

st.header("⚡ Live Prediction Results")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Response Time", f"{prediction:.2f} ms")
col2.metric("Model R² (Accuracy)", f"{r2:.4f}")
col3.metric("Prediction RMSE (Error)", f"{rmse:.2f} ms")

st.divider()

# --- LIVE VISUALIZATIONS ---
st.header("📈 Live Model Visualizations")
st.markdown("As you move the sliders in the sidebar, watch the **Red Star (★)** jump to see exactly where your simulated server sits compared to historical data!")

tab1, tab2, tab3 = st.tabs(["Live CPU Impact", "Live Network Impact", "Statistical Validation (Unit V)"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(8,4))
    # Historical Background
    ax1.scatter(df['CPU_Usage_Pct'], df['Response_Time_ms'], alpha=0.2, color='darkgreen', label='Historical Observations')
    # Live Point
    ax1.scatter([cpu_usage], [prediction], color='red', marker='*', s=400, edgecolor='black', label='Your Live Prediction')
    
    ax1.set_title("CPU Usage vs Response Time")
    ax1.set_xlabel("CPU Usage (%)")
    ax1.set_ylabel("Response Time (ms)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(8,4))
    # Historical Background
    ax2.scatter(df['Network_Load_MBps'], df['Response_Time_ms'], alpha=0.2, color='royalblue', label='Historical Observations')
    # Live Point
    ax2.scatter([net_load], [prediction], color='red', marker='*', s=400, edgecolor='black', label='Your Live Prediction')
    
    ax2.set_title("Network Load vs Response Time")
    ax2.set_xlabel("Network Load (MB/s)")
    ax2.set_ylabel("Response Time (ms)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig2)

with tab3:
    st.markdown("**Student's T-Test on Residuals ($H_0$: Mean error = 0)**")
    st.write(f"- **t-statistic:** `{t_stat:.4f}`")
    st.write(f"- **p-value:** `{p_value:.4f}`")
    if p_value > 0.05:
        st.success("✅ **Conclusion:** Fail to reject $H_0$. The model errors are statistically centered around zero (unbiased).")
    else:
        st.error("❌ **Conclusion:** Reject $H_0$. The model exhibits systematic bias.")
    
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.hist(residuals, bins=25, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Mean=0 (Expected)')
    ax3.axvline(residuals.mean(), color='yellow', linestyle='dashed', linewidth=2, label=f'Actual Mean={residuals.mean():.2f}')
    ax3.set_title("Distribution of Residuals")
    ax3.set_xlabel("Prediction Error (ms)")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig3)
