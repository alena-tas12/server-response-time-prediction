import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import os

def generate_server_data(n_samples=1000):
    """Generates a synthetic server performance dataset as a time-series."""
    np.random.seed(42)
    time_steps = np.arange(n_samples)
    
    # Workload parameters (Independent Variables)
    request_rate = np.random.randint(50, 1000, n_samples)
    cpu_usage = np.clip(request_rate * 0.08 + np.random.normal(0, 10, n_samples), 1, 100)
    memory_usage = np.clip(cpu_usage * 0.5 + 20 + np.random.normal(0, 5, n_samples), 1, 100)
    network_load = np.clip(request_rate * 0.05 + np.random.normal(0, 2, n_samples), 1, 100)
    
    # Introduce a slight statistical trend over time (Unit IV: Time-Series Components)
    trend = time_steps * 0.05 
    
    # Target variable (Dependent Variable): Response Time
    response_time = 50 + (cpu_usage * 2.5) + (memory_usage * 1.5) + (network_load * 3.0) + trend + np.random.normal(0, 20, n_samples)
    
    df = pd.DataFrame({
        'Time_Step': time_steps,
        'Request_Rate': request_rate,
        'CPU_Usage_Pct': cpu_usage,
        'Memory_Usage_Pct': memory_usage,
        'Network_Load_MBps': network_load,
        'Response_Time_ms': response_time
    })
    return df

def main():
    print("======================================================")
    print(" STATISTICAL SERVER RESPONSE TIME PREDICTION PIPELINE")
    print(" (Aligned with Syllabus Units IV & V)")
    print("======================================================\n")

    # 1. DATA COLLECTION
    print("[1] Collecting Performance Observations (Time-Series Data)...")
    df = generate_server_data(1000)
    df.to_csv('server_performance_data.csv', index=False)
    
    # 2. DESCRIPTIVE STATISTICS
    print("\n[2] Descriptive Statistics (Unit IV)...")
    print(df[['Request_Rate', 'CPU_Usage_Pct', 'Response_Time_ms']].describe().T[['mean', 'std', 'min', 'max']])
    
    # 3. CORRELATION ANALYSIS
    print("\n[3] Correlation Analysis (Unit IV)...")
    corr_matrix = df.corr()
    print("    Correlation with Response Time:")
    print(corr_matrix['Response_Time_ms'].sort_values(ascending=False).to_string())
    
    # 4. REGRESSION MODELLING
    print("\n[4] Developing Regression Model (Unit IV)...")
    features = ['Request_Rate', 'CPU_Usage_Pct', 'Memory_Usage_Pct', 'Network_Load_MBps']
    X = df[features]
    y = df['Response_Time_ms']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("    -> Regression Coefficients (Statistical Justification - CO4):")
    for feature, coef in zip(features, model.coef_):
        print(f"       {feature}: {coef:.3f}")
        
    # 5. PREDICTION & ERROR METRICS
    print("\n[5] Response Time Prediction & Error Evaluation (Unit IV)...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"    -> Mean Absolute Error (MAE): {mae:.2f} ms")
    print(f"    -> Root Mean Sq Error (RMSE): {rmse:.2f} ms")
    print(f"    -> Coefficient of Det. (R):  {r2:.4f}")
    
    # 6. STATISTICAL VALIDATION (HYPOTHESIS TESTING)
    print("\n[6] Statistical Validation / Hypothesis Testing (Unit V, CO5)...")
    residuals = y_test - y_pred
    
    # One-sample t-test on residuals to check if the model is unbiased
    # Null hypothesis (H0): The mean of the prediction errors is 0.
    t_stat, p_value = stats.ttest_1samp(residuals, 0)
    
    print("    -> Student's T-Test on Residuals (H0: Mean error = 0)")
    print(f"       t-statistic: {t_stat:.4f}")
    print(f"       p-value:     {p_value:.4f}")
    
    if p_value > 0.05:
        print("       Conclusion: Fail to reject H0. The model errors are statistically centered around zero (unbiased).")
    else:
        print("       Conclusion: Reject H0. The model exhibits systematic bias.")

    # 7. STATISTICAL VISUALIZATION
    print("\n[7] Generating Statistical Visualizations...")
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Time-Series Trend (Unit IV)
    plt.figure(figsize=(10,5))
    plt.plot(df['Time_Step'][:200], df['Response_Time_ms'][:200], alpha=0.8, color='steelblue')
    plt.title('Time-Series Trend of Response Time (First 200 obs) [Unit IV]')
    plt.xlabel('Time Step')
    plt.ylabel('Response Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('plots/1_time_series_trend.png')
    plt.close()
    
    # Plot 2: Correlation Scatter (Unit IV)
    plt.figure(figsize=(8,6))
    plt.scatter(df['CPU_Usage_Pct'], df['Response_Time_ms'], alpha=0.5, color='darkgreen')
    plt.title('Correlation: CPU Usage vs Response Time [Unit IV]')
    plt.xlabel('CPU Usage (%)')
    plt.ylabel('Response Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('plots/2_correlation_scatter.png')
    plt.close()
    
    # Plot 3: Residual Distribution for Hypothesis Testing (Unit V)
    plt.figure(figsize=(8,6))
    plt.hist(residuals, bins=25, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Mean=0 (Expected)')
    plt.axvline(residuals.mean(), color='yellow', linestyle='dashed', linewidth=2, label=f'Actual Mean={residuals.mean():.2f}')
    plt.title('Statistical Validation: Distribution of Residuals [Unit V]')
    plt.xlabel('Prediction Error (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('plots/3_residual_distribution.png')
    plt.close()
    
    print("    -> Visualizations saved to 'plots/' directory.")
    print("\nSYSTEM PIPELINE COMPLETE.")

if __name__ == "__main__":
    main()
