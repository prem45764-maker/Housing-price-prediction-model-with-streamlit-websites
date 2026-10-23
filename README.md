# Housing-price-prediction-model-with-streamlit-websites
# 🏠 Ames Housing Price Prediction & Interactive Streamlit Dashboard

An end-to-end Machine Learning web application and high-precision regression pipeline designed to predict real estate property sale prices with exceptional accuracy.

## 📊 Model Performance & Metrics
- **R² Score:** ~93.7% - 95.3%
- **Root Mean Squared Error (RMSE):** ~$18,000 - $20,000
- **Average Error Percentage:** ~5.56%

## 🚀 Key Features
- **Advanced Feature Engineering:** Custom features created for property square footage, total baths, house age, remodeling status, and quality-area interactions.
- **Log-Transformation:** Applied `np.log1p` on the target variable (`SalePrice`) to normalize right-skewed real estate data and boost model stability.
- **Robust Preprocessing Pipeline:** Implemented Scikit-Learn `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, `OrdinalEncoder`, and `OneHotEncoder` to handle missing values and categorical features cleanly.
- **Interactive Streamlit Web App:** A clean, user-friendly dashboard to visualize model performance, real vs. predicted scatter plots, and live error metrics instantly.

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.x
- **Machine Learning & Preprocessing:** Scikit-Learn, XGBoost, Gradient Boosting, CatBoost
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Web App Framework:** Streamlit
- **Model Serialization:** Joblib

## ⚙️ How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/Housing-price-prediction-model-with-streamlit-websites.git](https://github.com/your-username/Housing-price-prediction-model-with-streamlit-websites.git)
   cd Housing-price-prediction-model-with-streamlit-websites
