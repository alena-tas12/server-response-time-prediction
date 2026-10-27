# Server Response Time Prediction Using Regression Models

This repository contains the complete implementation for **Micro-Project 13: Server Response Time Prediction Using Regression Models**, rigorously aligned with the **Probability and Statistics for Data Science and Cyber Security Engineers (25DC210)** syllabus.

## 1.0 RATIONALE

Modern web services operate under continuous conditions of uncertainty. The response time observed by a client is not a fixed mathematical constant; it is a random variable that fluctuates heavily based on uncertain workload and server-performance parameters (e.g., request rate, CPU utilization, network activity). 

The purpose of this project is to apply statistical modelling to understand, model, and predict this uncertainty. Rather than treating this purely as an artificial intelligence or software engineering problem, this system approaches response time fundamentally through the lens of statistics. By analyzing historical performance observations, we can identify correlations between workload parameters and response delays, apply regression analysis to model these relationships, and statistically validate the accuracy of those predictions using hypothesis testing and error measurement.

This establishes a computational pipeline built on statistics:
**Uncertainty → Statistical Modelling → Correlation → Regression → Prediction → Validation**

## 2.0 PROPOSED METHODOLOGY

This project directly implements the statistical methodologies from the course syllabus (Units I–V):

1. **Collect performance observations:** Simulate/capture uncertain workload and response-time variables.
2. **Represent variables:** Structure the data for random variable analysis.
3. **Analyze descriptive statistics:** Compute means, variances, and bounds.
4. **Examine trends:** Analyze the time-series components of response times *(Unit IV)*.
5. **Calculate correlation:** Measure the statistical correlation between workload and response *(Unit IV)*.
6. **Develop regression model:** Fit a Multiple Linear Regression equation to the variables *(Unit IV)*.
7. **Predict response time:** Estimate future outcomes under uncertain network conditions *(Unit III/IV)*.
8. **Evaluate prediction error:** Quantify accuracy using MAE, RMSE, and $R^2$.
9. **Statistically validate the model:** Perform hypothesis testing (e.g., Student's t-test on residuals) to prove model unbiasedness *(Unit V)*.
10. **Interpret results:** Relate the statistical findings back to the web-service domain.

## 3.0 COURSE OUTCOME MAPPING

This project structurally fulfills the prescribed Course Outcomes (COs):

| CO | How Project 13 addresses it |
| :--- | :--- |
| **CO2** | Model the uncertainty and variation in server response time using workload-related data distributions. |
| **CO3** | Predict response time and network-related performance under varying, uncertain workload conditions using regression. |
| **CO4** | Statistically justify the relationship between workload variables and predicted response time using correlation and regression coefficients. |
| **CO5** | Validate the regression model using statistical error measures (RMSE, $R^2$) and hypothesis-testing techniques (Student's t-test on residuals). |

## 4.0 RUNNING THE PROJECT

### Prerequisites
Install the required statistical dependencies:
```bash
pip install -r requirements.txt
```

### Execution
Run the statistical pipeline script:
```bash
python server_prediction.py
```

Or launch the interactive Streamlit Web Application:
```bash
streamlit run app.py
```

### 🌐 Live Future Roadmap Deck
The interactive presentation deck is deployed and hosted on GitHub Pages:
👉 **[View Future Roadmap: Latency Prediction to Cyber Risk Scoring](https://alena-tas12.github.io/server-response-time-prediction/)**

### Output
- `server_performance_data.csv`: The underlying simulated observation dataset.
- `plots/`: Contains statistical visualizations (Time-Series Trend, Correlation Scatter, Residual Distribution).
- Console output detailing descriptive statistics, correlation coefficients, regression weights, and the residual T-Test results.
