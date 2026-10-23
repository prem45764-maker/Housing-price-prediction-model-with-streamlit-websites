import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
df = pd.read_csv("AmesHousing.csv")
print(df.head(4))
print(df.info())
df = df.drop_duplicates()
for i in df.select_dtypes(include=["int64","float64"]):
    print(f"-----{i}-----")
    print(df[i].max())
    print(df[i].min())
for i in df.select_dtypes(include=["object"]):
    print(f"-----{i}-----")
    print(df[i].unique())
df = df[df["Gr Liv Area"] < 4000].reset_index(drop=True)

df.drop(columns=["Order", "PID"], errors="ignore", inplace=True)
df["MS SubClass"] = df["MS SubClass"].astype(str)

# ==========================================
# STEP 2: Advanced Feature Engineering
# ==========================================
df["House_Age"] = df["Yr Sold"] - df["Year Built"]
df["Remod_Age"] = df["Yr Sold"] - df["Year Remod/Add"]
df["Is_Remodeled"] = (df["Year Remod/Add"] != df["Year Built"]).astype(int)

df["Total_SF"] = (
    df["1st Flr SF"].fillna(0)
    + df["2nd Flr SF"].fillna(0)
    + df["Total Bsmt SF"].fillna(0)
)
df["Area_x_Quality"] = df["Total_SF"] * df["Overall Qual"]

df["Total_Baths"] = (
    df["Full Bath"].fillna(0)
    + (0.5 * df["Half Bath"].fillna(0))
    + df["Bsmt Full Bath"].fillna(0)
    + (0.5 * df["Bsmt Half Bath"].fillna(0))
)

df["Total_Porch_SF"] = (
    df["Wood Deck SF"].fillna(0)
    + df["Open Porch SF"].fillna(0)
    + df["Enclosed Porch"].fillna(0)
    + df["3Ssn Porch"].fillna(0)
    + df["Screen Porch"].fillna(0)
)

df["Has_Garage"] = (df["Garage Area"].fillna(0) > 0).astype(int)
df["Has_Bsmt"] = (df["Total Bsmt SF"].fillna(0) > 0).astype(int)
df["Has_Fireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)

# Features & Target
X = df.drop(columns=["SalePrice"])

# 🎯 CRITICAL ACCURACY BOOST: Log-Transform Target Variable
# Real estate prices right-skewed hoti hain. Log-transforming standardizes variance.
y = np.log1p(df["SalePrice"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# STEP 3: Column Lists Definition
# ==========================================
numerical_cols = [
    "Lot Frontage",
    "Lot Area",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Yr Blt",
    "Garage Cars",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Misc Val",
    "Mo Sold",
    "Yr Sold",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "House_Age",
    "Remod_Age",
    "Is_Remodeled",
    "Total_SF",
    "Area_x_Quality",
    "Total_Baths",
    "Total_Porch_SF",
    "Has_Garage",
    "Has_Bsmt",
    "Has_Fireplace",
]

ordinal_cols = [
    "Lot Shape",
    "Utilities",
    "Land Slope",
    "Exter Qual",
    "Exter Cond",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Heating QC",
    "Electrical",
    "Kitchen Qual",
    "Functional",
    "Fireplace Qu",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Paved Drive",
    "Pool QC",
    "Fence",
]

nominal_cols = [
    "MS SubClass",
    "MS Zoning",
    "Street",
    "Alley",
    "Land Contour",
    "Lot Config",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Type",
    "Foundation",
    "Heating",
    "Central Air",
    "Garage Type",
    "Misc Feature",
    "Sale Type",
    "Sale Condition",
]

# ==========================================
# STEP 4: Preprocessing Pipelines
# ==========================================
qual_cond_order = ["Missing", "Po", "Fa", "TA", "Gd", "Ex"]
lot_shape_order = ["IR3", "IR2", "IR1", "Reg"]
util_order = ["Elo", "NoSeWa", "NoSewr", "AllPub"]
land_slope_order = ["Sev", "Mod", "Gtl"]
bsmt_exp_order = ["Missing", "No", "Mn", "Av", "Gd"]
bsmt_fin_order = ["Missing", "Unf", "LwQ", "REC", "BLQ", "ALQ", "GLQ"]
heating_qc_order = ["Missing", "Po", "Fa", "TA", "Gd", "Ex"]
elect_order = ["Missing", "Mix", "FuseP", "FuseF", "FuseA", "SBrkr"]
func_order = ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]
gar_fin_order = ["Missing", "Unf", "RFn", "Fin"]
paved_order = ["N", "P", "Y"]
fence_order = ["Missing", "MnWw", "GdWo", "MnPrv", "GdPrv"]

categories_list = [
    lot_shape_order,
    util_order,
    land_slope_order,
    qual_cond_order,
    qual_cond_order,
    qual_cond_order,
    qual_cond_order,
    bsmt_exp_order,
    bsmt_fin_order,
    bsmt_fin_order,
    heating_qc_order,
    elect_order,
    qual_cond_order,
    func_order,
    qual_cond_order,
    gar_fin_order,
    qual_cond_order,
    qual_cond_order,
    paved_order,
    qual_cond_order,
    fence_order,
]

numerical_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OrdinalEncoder(
            # Agar categories mein confusion ho rahi hai toh 'auto' use kar lo,
            # ya fir categories wali list ka size match karo.
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
    ),
])

nominal_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("ordinal", ordinal_pipeline, ordinal_cols),
        ("nominal", nominal_pipeline, nominal_cols),
    ]
)

# ==========================================
# STEP 5: Optimized Models Setup
# ==========================================
# Tuned Hyperparameters for >90% Performance
pipe_gbr = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=4,
                max_features="sqrt",
                subsample=0.8,
                random_state=42,
            ),
        ),
    ]
)

# ==========================================
# STEP 6: Training & True Dollar Evaluation
# ==========================================
print("🔥 Training High-Precision Models...\n" + "=" * 55)

pipe_gbr.fit(X_train, y_train)

# Predictions (In Log Scale)
preds_log = pipe_gbr.predict(X_test)
import numpy as np
# 🎯 Convert Log Predictions back to Actual Dollars ($)
preds_dollars = np.expm1(preds_log)
y_test_dollars = np.expm1(y_test)

# Evaluation on True Dollars
print("Test Score:" ,pipe_gbr.score(X_test,y_test)*100)
print("Train Score:" ,pipe_gbr.score(X_train,y_train)*100)
r2 = r2_score(y_test_dollars, preds_dollars)
rmse = np.sqrt(mean_squared_error(y_test_dollars, preds_dollars))
print("🎉R2 score:" ,r2 * 100)
print(f"💵 Real Dollar RMSE: ${rmse:,.2f}")
print("=" * 45)
# 2. Ab safely model save kar
import joblib

joblib.dump(pipe_gbr, "housing_price_gbmboost_pipeline.pkl")

print("Model successfully save ho gaya bhai!")
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test_dollars, preds_dollars) * 100
print(f"📊 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# Actual minus Predicted Dollars
errors = y_test_dollars - preds_dollars

mean_error = np.mean(errors)
print(f"Mean Error ($): {mean_error:,.2f}")

if mean_error > 0:
    print("👉 Model overall THODA KAM (Underestimate) price bata raha hai (Safe for Buyers).")
elif mean_error < 0:
    print("👉 Model overall THODA ZYADA (Overestimate) price bata raha hai (Good for Sellers).")
else:
    print("👉 Model completely Perfectly Balanced hai!")
joblib.dump(pipe_gbr, "housing_price_gbmboost_pipeline.pkl")
df.to_csv("test1.csv")
feature_names = pipe_gbr.named_steps[
    "preprocessor"  # <-- 'P' ko 'p' kar diya
].get_feature_names_out()

importances = pipe_gbr.named_steps["model"].feature_importances_

feat_imp_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values(by="Importance", ascending=False)

print("\n--- Top Features Driving the Business Model ---")
print(feat_imp_df.head(10))
df1 = feat_imp_df
print(df1.head(2))
# 0.01% matlab 0.0001 (kyunki importances 0 se 1 ke scale par hoti hain)
threshold = 0.01

filtered_feat_imp = feat_imp_df[feat_imp_df["Importance"] > threshold]

print(f"\n--- Features with > 0.01% Importance ({len(filtered_feat_imp)} features) ---")
print(filtered_feat_imp.head(12))