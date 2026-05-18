# Glossary

Terms used across the notebooks and README for this project.

---

## Clinical / CGM terms

| acronym | full term | meaning |
|---|---|---|
| CGM | Continuous Glucose Monitor | A wearable sensor that measures interstitial glucose every few minutes without fingersticks |
| TIR | Time In Range | Percentage of CGM readings between 70 and 180 mg/dL; the standard target is >70% |
| GMI | Glucose Management Indicator | An estimated HbA1c derived from CGM mean glucose: `3.31 + 0.02392 × mean_mg_dL` (Bergenstal 2018) |
| A1C / HbA1c | Hemoglobin A1c | A lab test measuring average blood glucose over the past ~3 months; target for well-controlled T2D is <7% |
| IF | Intermittent Fasting | An eating pattern that restricts food to a defined window; here used to mean skipping breakfast and eating first at lunch |
| mg/dL | milligrams per deciliter | The glucose concentration unit used in the US |
| peak_delta | — | The rise in glucose from pre-meal baseline to peak within ~3 hours; the primary modeling target |

---

## Machine learning / statistics terms

| acronym | full term | meaning |
|---|---|---|
| EDA | Exploratory Data Analysis | Initial investigation of a dataset through summary statistics and visualizations |
| RMSE | Root Mean Squared Error | Square root of the average squared prediction error; penalises large misses more than MAE |
| MAE | Mean Absolute Error | Average absolute difference between predicted and actual values, in the original unit (mg/dL here) |
| R² | Coefficient of determination | Proportion of variance in the target explained by the model; 1.0 is perfect, 0 means the model does no better than predicting the mean |
| SHAP | SHapley Additive exPlanations | A method from cooperative game theory that assigns each feature a contribution value for each individual prediction |
| GBM | Gradient Boosting Machine | An ensemble of decision trees built sequentially; here specifically `HistGradientBoostingRegressor` from scikit-learn |
| SARIMAX | Seasonal AutoRegressive Integrated Moving Average with eXogenous variables | A time-series forecasting model that accounts for trend, seasonality, autocorrelation, and external regressors |
| ACF | AutoCorrelation Function | Measures correlation between a time series and its own lagged values; used to identify the MA order |
| PACF | Partial AutoCorrelation Function | Like ACF but removes indirect correlations through intermediate lags; used to identify the AR order |
| ADF | Augmented Dickey-Fuller test | A statistical test for whether a time series is stationary (no unit root); p < 0.05 means stationary |
| DW | Durbin-Watson statistic | Tests for autocorrelation in regression residuals; ~2.0 means none, below ~1.0 means severe positive autocorrelation (OLS assumptions violated) |
| AR / MA | AutoRegressive / Moving Average | The two core components of ARIMA: AR uses past values, MA uses past forecast errors |
| oracle vs recursive | — | Oracle evaluation feeds a model the true recent value as an input (optimistic); recursive evaluation feeds the model its own prior prediction (realistic for forecasting ahead) |
| CV | Cross-Validation | Technique for estimating model performance by splitting data into multiple train/validation folds |
| DOW | Day of Week | Calendar day Monday–Sunday |
