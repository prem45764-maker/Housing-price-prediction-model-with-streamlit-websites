import streamlit as st 
import joblib 
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="💰",
    layout="wide"
)

@st.cache_data
def load_model():
    return joblib.load("housing_price_gbmboost_pipeline.pkl")
    
model = load_model()

st.title("💰 Smart House Price Prediction Engine")
st.write("Analyze individual profiles or bulk process datasets with automated AI backend optimization.")

def feature_engineering(df):
    df = df.copy()
    df.drop(columns=["Order", "PID"], errors="ignore", inplace=True)
    if "MS SubClass" in df.columns:
        df["MS SubClass"] = df["MS SubClass"].astype(str)

    # Advanced Feature Engineering
    if "Yr Sold" in df.columns and "Year Built" in df.columns:
        df["House_Age"] = df["Yr Sold"] - df["Year Built"]
    if "Yr Sold" in df.columns and "Year Remod/Add" in df.columns:
        df["Remod_Age"] = df["Yr Sold"] - df["Year Remod/Add"]
        df["Is_Remodeled"] = (df["Year Remod/Add"] != df["Year Built"]).astype(int)

    if all(col in df.columns for col in ["1st Flr SF", "2nd Flr SF", "Total Bsmt SF"]):
        df["Total_SF"] = (
            df["1st Flr SF"].fillna(0)
            + df["2nd Flr SF"].fillna(0)
            + df["Total Bsmt SF"].fillna(0)
        )
    if "Total_SF" in df.columns and "Overall Qual" in df.columns:
        df["Area_x_Quality"] = df["Total_SF"] * df["Overall Qual"]

    if all(col in df.columns for col in ["Full Bath", "Half Bath", "Bsmt Full Bath", "Bsmt Half Bath"]):
        df["Total_Baths"] = (
            df["Full Bath"].fillna(0)
            + (0.5 * df["Half Bath"].fillna(0))
            + df["Bsmt Full Bath"].fillna(0)
            + (0.5 * df["Bsmt Half Bath"].fillna(0))
        )

    if all(col in df.columns for col in ["Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"]):
        df["Total_Porch_SF"] = (
            df["Wood Deck SF"].fillna(0)
            + df["Open Porch SF"].fillna(0)
            + df["Enclosed Porch"].fillna(0)
            + df["3Ssn Porch"].fillna(0)
            + df["Screen Porch"].fillna(0)
        )

    if "Garage Area" in df.columns:
        df["Has_Garage"] = (df["Garage Area"].fillna(0) > 0).astype(int)
    if "Total Bsmt SF" in df.columns:
        df["Has_Bsmt"] = (df["Total Bsmt SF"].fillna(0) > 0).astype(int)
    if "Fireplaces" in df.columns:
        df["Has_Fireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)

    return df

# ==========================================
# SIDEBAR INPUTS (Top High-Importance Features)
# ==========================================
st.sidebar.header("🏠 Property Configuration")
st.sidebar.subheader("Enter core high-importance property details.")

neighborhood = st.sidebar.selectbox(
    "Neighborhood", 
    ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'NridgHt', 'Gilbert', 
     'Sawyer', 'SawyerW', 'NWAmes', 'Mitchel', 'BrkSide', 'Crawfor', 'IDOTRR', 
     'Timber', 'NoRidge', 'StoneBr', 'SWISU', 'ClearCr', 'MeadowV', 'BrDale', 
     'Blmngtn', 'Veenker', 'NPkVill', 'Blueste', 'Greens', 'GrnHill', 'Landmrk'],
    index=0
)

bldg_type = st.sidebar.selectbox(
    "Building Type", 
    ['1Fam', 'TwnhsE', 'Twnhs', 'Duplex', '2fmCon'],
    index=0
)

exter_qual = st.sidebar.selectbox(
    "Exterior Quality", 
    ['TA', 'Gd', 'Ex', 'Fa'],
    index=0
)

kitchen_qual = st.sidebar.selectbox(
    "Kitchen Quality", 
    ['TA', 'Gd', 'Ex', 'Fa', 'Po'],
    index=0
)

foundation = st.sidebar.selectbox(
    "Foundation Type", 
    ['PConc', 'CBlock', 'BrkTil', 'Slab', 'Stone', 'Wood'],
    index=0
)

central_air = st.sidebar.selectbox("Central Air", ['Y', 'N'], index=0)
garage_finish = st.sidebar.selectbox("Garage Finish", ['Unf', 'RFn', 'Fin'], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Key Numerical Drivers")

overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
overall_cond = st.sidebar.slider("Overall Condition (1-9)", 1, 9, 5)
gr_liv_area = st.sidebar.slider("Above Ground Living Area (sq ft)", 334, 3820, 1500)
total_bsmt_sf = st.sidebar.slider("Total Basement SF", 0, 3200, 800)
first_flr_sf = st.sidebar.slider("1st Floor SF", 334, 3820, 1000)
second_flr_sf = st.sidebar.slider("2nd Floor SF", 0, 1862, 500)
lot_area = st.sidebar.slider("Lot Area (sq ft)", 1300, 215245, 9430)
garage_cars = st.sidebar.slider("Garage Car Capacity", 0, 5, 2)
garage_area = st.sidebar.slider("Garage Area (sq ft)", 0, 1488, 480)
fireplaces = st.sidebar.slider("Fireplaces", 0, 4, 1)
total_baths = st.sidebar.slider("Total Bathrooms", 1.0, 7.0, 2.0, step=0.5)
year_built = st.sidebar.slider("Year Built", 1872, 2010, 2005)
year_remod = st.sidebar.slider("Year Remod/Add", 1950, 2010, 2005)

# --- TABS FOR SINGLE & BULK AUDIT ---
tab1, tab2 = st.tabs(["👤 Single Property Audit", "📁 Bulk CSV Batch Audit"])

# ==========================================
# TAB 1: SINGLE PROPERTY AUDIT
# ==========================================
with tab1:
    st.subheader("👤 Single Property Prediction Dashboard")
    st.write("Sidebar se property ki details enter karo aur neeche button dabakar price predict karo.")

    if st.button("🚀 Predict Single House Price"):
        input_data = {
            'Neighborhood': neighborhood,
            'Bldg Type': bldg_type,
            'Exter Qual': exter_qual,
            'Kitchen Qual': kitchen_qual,
            'Foundation': foundation,
            'Central Air': central_air,
            'Garage Finish': garage_finish,
            'Overall Qual': overall_qual,
            'Overall Cond': overall_cond,
            'Gr Liv Area': gr_liv_area,
            'Total Bsmt SF': total_bsmt_sf,
            '1st Flr SF': first_flr_sf,
            '2nd Flr SF': second_flr_sf,
            'Lot Area': lot_area,
            'Garage Cars': garage_cars,
            'Garage Area': garage_area,
            'Fireplaces': fireplaces,
            'Full Bath': int(total_baths),
            'Half Bath': 1 if (total_baths % 1 != 0) else 0,
            'Year Built': year_built,
            'Year Remod/Add': year_remod,
            
            # Smart Defaults for remaining secondary columns
            'MS SubClass': '20',
            'MS Zoning': 'RL',
            'Street': 'Pave',
            'Alley': 'NA',
            'Lot Shape': 'Reg',
            'Land Contour': 'Lvl',
            'Utilities': 'AllPub',
            'Lot Config': 'Inside',
            'Land Slope': 'Gtl',
            'Condition 1': 'Norm',
            'Condition 2': 'Norm',
            'House Style': '1Story',
            'Roof Style': 'Gable',
            'Roof Matl': 'CompShg',
            'Exterior 1st': 'VinylSd',
            'Exterior 2nd': 'VinylSd',
            'Mas Vnr Type': 'None',
            'Exter Cond': 'TA',
            'Bsmt Qual': 'TA',
            'Bsmt Cond': 'TA',
            'Bsmt Exposure': 'No',
            'BsmtFin Type 1': 'Unf',
            'BsmtFin Type 2': 'Unf',
            'Heating': 'GasA',
            'Heating QC': 'Ex',
            'Electrical': 'SBrkr',
            'Functional': 'Typ',
            'Fireplace Qu': 'NA',
            'Garage Type': 'Attchd',
            'Garage Qual': 'TA',
            'Garage Cond': 'TA',
            'Paved Drive': 'Y',
            'Pool QC': 'NA',
            'Fence': 'NA',
            'Misc Feature': 'NA',
            'Sale Type': 'WD ',
            'Sale Condition': 'Normal',
            'Lot Frontage': 68.0,
            'Mas Vnr Area': 0.0,
            'BsmtFin SF 1': 368.0,
            'BsmtFin SF 2': 0.0,
            'Bsmt Unf SF': 467.0,
            'Low Qual Fin SF': 0.0,
            'Bsmt Full Bath': 0.0,
            'Bsmt Half Bath': 0.0,
            'Bedroom AbvGr': 3.0,
            'Kitchen AbvGr': 1.0,
            'TotRms AbvGrd': 6.0,
            'Garage Yr Blt': float(year_built),
            'Wood Deck SF': 0.0,
            'Open Porch SF': 40.0,
            'Enclosed Porch': 0.0,
            '3Ssn Porch': 0.0,
            'Screen Porch': 0.0,
            'Pool Area': 0.0,
            'Misc Val': 0.0,
            'Mo Sold': 6.0,
            'Yr Sold': 2010
        }

        df_input = pd.DataFrame([input_data])
        df_processed = feature_engineering(df_input)
        # Tab 1 ke andar prediction line aisi honi chahiye:
        raw_pred = model.predict(df_processed)[0]
        prediction = np.expm1(raw_pred)  # 👈 Log prediction ko real dollar ($) mein badalne ke liye

        st.success(f"### Predicted House Price: ${prediction:,.2f} 💰")
# ==========================================
# TAB 2: BULK CSV BATCH AUDIT WITH METRICS
# ==========================================
# ==========================================
# TAB 2: BULK CSV BATCH AUDIT WITH DETAILED COMPARISON
# ==========================================
# ==========================================
# TAB 2: BULK CSV BATCH AUDIT WITH GRAPH & METRICS
# ==========================================
with tab2:
    st.subheader("📁 Upload CSV Dataset for Detailed Price Comparison")
    st.write("Apni CSV file upload karo jisme real `SalePrice` ho. Model real price aur predicted price ko compare karke dikhayega ki kitni galti ho rahi hai.")

    uploaded_file = st.file_uploader("Choose a CSV file for comparison", type=["csv"], key="bulk_eval")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("🔍 **Uploaded Data Preview:**")
        st.dataframe(batch_df.head())

        if st.button("⚙️ Run Comparison & Analysis"):
            try:
                if "SalePrice" not in batch_df.columns:
                    st.error("⚠️ Error: Uploaded CSV file mein 'SalePrice' column nahi mila! Comparison ke liye actual SalePrice hona zaroori hai.")
                else:
                    # 1. Feature engineering aur predictions (Log scale to real dollars)
                    batch_processed = feature_engineering(batch_df)
                    raw_preds = model.predict(batch_processed)
                    batch_predictions = np.expm1(raw_preds)  
                    
                    # 2. Results DataFrame banana
                    comparison_df = batch_df.copy()
                    comparison_df["Predicted_SalePrice"] = batch_predictions
                    
                    # 3. Error aur Difference calculate karna
                    comparison_df["Price_Difference (\$)"] = comparison_df["Predicted_SalePrice"] - comparison_df["SalePrice"]
                    comparison_df["Absolute_Error (\$)"] = abs(comparison_df["Price_Difference (\$)"])
                    comparison_df["Error_Percentage (%)"] = (comparison_df["Absolute_Error (\$)"] / comparison_df["SalePrice"]) * 100

                    # Overall Metrics
                    r2 = r2_score(comparison_df["SalePrice"], comparison_df["Predicted_SalePrice"])
                    rmse = np.sqrt(mean_squared_error(comparison_df["SalePrice"], comparison_df["Predicted_SalePrice"]))
                    avg_error_dollar = comparison_df["Absolute_Error (\$)"].mean()
                    avg_error_percentage = comparison_df["Error_Percentage (%)"].mean()

                    st.markdown("---")
                    st.subheader("📊 Model vs Real Price Performance Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🎉 R2 Score", f"{r2 * 100:.2f}%")
                    col2.metric("💵 Root Mean Squared Error", f"\${rmse:,.2f}")
                    col3.metric("📉 Avg. Dollar Mistake", f"\${avg_error_dollar:,.2f}")
                    col4.metric("📈 Avg. Error Percentage", f"{avg_error_percentage:.2f}%")

                    # ==========================================
                    # 📊 ADDED GRAPH: Actual vs Predicted Scatter Plot
                    # ==========================================
                    st.markdown("---")
                    st.subheader("📈 Real vs Predicted Price Scatter Plot")
                    st.write("Red line 'Perfect Fit' line hai. Jitne dots is line ke paas honge, model utna hi accurate hoga.")
                    
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(
                        x=comparison_df["SalePrice"], 
                        y=comparison_df["Predicted_SalePrice"], 
                        alpha=0.7, 
                        ax=ax, 
                        color="blue"
                    )
                    # Perfect Fit Line (45-degree line)
                    min_val = min(comparison_df["SalePrice"].min(), comparison_df["Predicted_SalePrice"].min())
                    max_val = max(comparison_df["SalePrice"].max(), comparison_df["Predicted_SalePrice"].max())
                    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label="Perfect Fit Line")
                    
                    ax.set_xlabel("Actual Sale Price (\$)")
                    ax.set_ylabel("Predicted Sale Price (\$)")
                    ax.legend()
                    st.pyplot(fig)

                    st.markdown("---")
                    st.subheader("🔍 Detailed Property-by-Property Comparison Table")
                    st.write("Yahan dekh sakte ho ki har ek ghar ke liye real price kya tha aur model ne kitna predict kiya:")
                    
                    # Display table highlighting main comparison columns first
                    display_cols = ["SalePrice", "Predicted_SalePrice", "Price_Difference (\$)", "Error_Percentage (%)"]
                    other_cols = [c for c in comparison_df.columns if c not in display_cols]
                    
                    st.dataframe(comparison_df[display_cols + other_cols].head(20))

                    # Download button for detailed report
                    csv_output = comparison_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Detailed Comparison CSV",
                        data=csv_output,
                        file_name="house_price_detailed_comparison.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"⚠️ Error during evaluation: {e}")