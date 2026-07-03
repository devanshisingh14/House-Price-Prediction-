# ==========================================
# HOUSE PRICE PREDICTION - DATA PREPROCESSING
# ==========================================

import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plot  
import seaborn as sus  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== STEP 1: LOAD DATA =====
raw_data = pd.read_csv(r"C:\Users\Devanshi\Downloads\train.csv")
data = pd.DataFrame(raw_data)

print("\n" + "="*60)
print("STEP 1: CHECKING FOR MISSING VALUES")
print("="*60)

data = data[data['GrLivArea'] < 4000]
data = data[data['SalePrice'] < 700000]

# Log transform skewed features
data['SalePrice'] = np.log1p(data['SalePrice'])
data['LotArea'] = np.log1p(data['LotArea'])
data['GrLivArea'] = np.log1p(data['GrLivArea'])

print("\nMissing value count per column:")
print(data.isnull().sum())

# ===== STEP 2: REMOVE COLUMNS WITH TOO MANY MISSING VALUES =====
data_null_clm = data.isnull().sum() / data.shape[0] * 100  
null20_clm_list = data_null_clm[data_null_clm > 20].index
data_drop_clm = data.drop(columns=null20_clm_list)

print(f"\nRemoved {len(null20_clm_list)} columns with >20% missing values")
print("Remaining missing values:")
print(data_drop_clm.isnull().sum())

# ===== STEP 3: SEPARATE DATA BY TYPE =====
data_object = data_drop_clm.select_dtypes(include=['object'])
data_numaric = data_drop_clm.select_dtypes(include=['int', 'float'])

# ===== STEP 4: HANDLE MISSING VALUES =====
for i in data_numaric.columns:
    data_numaric[i].fillna(data_numaric[i].median(), inplace=True)

for i in data_object.columns:
    data_object[i].fillna(data_object[i].mode()[0], inplace=True)

# ===== STEP 5: ENCODING =====
data_encoded = data_object.copy()

# Ordinal Encoding
ordinal_mappings = {
    'ExterQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'BsmtQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'BsmtCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4},  
    'BsmtExposure': {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},  
    'BsmtFinType1': {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},  
    'BsmtFinType2': {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},  
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
    'KitchenQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'Functional': {'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},  
    'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3}, 
    'GarageQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'GarageCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3},  
    'PavedDrive': {'N': 1, 'P': 2, 'Y': 3}  
}

for col, mapping in ordinal_mappings.items():
    if col in data_encoded.columns:
        data_encoded[col] = data_encoded[col].map(mapping)

# One-Hot Encoding
onehot_columns = [
    'MSZoning','Street','LotShape','LotConfig','LandContour','Utilities','BldgType',      
    'HouseStyle','RoofStyle','RoofMatl','Foundation','Heating','CentralAir','Electrical',   
    'GarageType','Condition1','Condition2','SaleType','SaleCondition'  
]
data_encoded = pd.get_dummies(data_encoded, columns=onehot_columns, drop_first=True, dtype='int')

# Target Encoding
if 'Neighborhood' in data_encoded.columns and 'SalePrice' in data_encoded.columns:
    neighborhood_encoding = data_encoded.groupby('Neighborhood')['SalePrice'].mean()
    data_encoded['Neighborhood'] = data_encoded['Neighborhood'].map(neighborhood_encoding)
else:
    neighborhood_counts = data_encoded['Neighborhood'].value_counts()
    data_encoded['Neighborhood'] = data_encoded['Neighborhood'].map(neighborhood_counts)

# Exterior Columns Rare Grouping
for col in ['Exterior1st', 'Exterior2nd']:
    if col in data_encoded.columns:
        value_counts = data_encoded[col].value_counts()
        rare_threshold = len(data_encoded) * 0.02
        rare_values = value_counts[value_counts < rare_threshold].index
        data_encoded[col] = data_encoded[col].replace(rare_values, 'Other')
        data_encoded = pd.get_dummies(data_encoded, columns=[col], prefix=col, drop_first=True, dtype='int')

# ===== FINAL DATASET =====
final_data = pd.concat([data_numaric, data_encoded], axis=1)
final_data = final_data.loc[:, ~final_data.columns.duplicated()]

# 🚨 FIX: Ensure no NaNs remain
final_data = final_data.fillna(0)

X = final_data.drop(columns=['SalePrice'])
y = final_data['SalePrice']

# ===== Train/Test Split =====
x_train, x_test, y_train_target, y_test_target = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ===== Scaling =====
scaler = StandardScaler()
scaled_data_final_X_tarin_arr = scaler.fit_transform(x_train)
scaled_data_final_X_tarin = pd.DataFrame(scaled_data_final_X_tarin_arr, columns=x_train.columns)

scaled_data_final_X_test_arr = scaler.transform(x_test)
scaled_data_final_X_test = pd.DataFrame(scaled_data_final_X_test_arr, columns=x_test.columns)

# ===== Model =====
lr = LinearRegression()
lr.fit(scaled_data_final_X_tarin, y_train_target)

Y = lr.predict(scaled_data_final_X_test)
print(Y)

# ===== Plots =====
fig, axs = plot.subplots(3, 1, figsize=(10, 15))
axs[0].scatter(scaled_data_final_X_tarin['GrLivArea'], y_train_target, marker='.', color='blue')
axs[0].set_title("Scatter plot: GrLivArea vs SalePrice")
axs[0].set_xlabel("GrLivArea (scaled)")
axs[0].set_ylabel("SalePrice (log)")

axs[1].plot(y_test_target.values, label="Actual", color="blue")
axs[1].plot(Y, label="Predicted", color="red", linestyle="--")
axs[1].legend()
axs[1].set_title("Actual vs Predicted SalePrice")

sorted_index = np.argsort(y_test_target.values)
y_test_sorted = y_test_target.values[sorted_index]
Y_sorted = Y[sorted_index]

axs[2].plot(y_test_sorted, label="Actual (sorted)", color="blue")
axs[2].plot(Y_sorted, label="Predicted (sorted)", color="red", linestyle="--")
axs[2].legend()
axs[2].set_title("Sorted Actual vs Predicted SalePrice")

plot.tight_layout()
plot.show()

# ===== Metrics =====
mse = mean_squared_error(y_test_target, Y)
mae = mean_absolute_error(y_test_target, Y)
r2 = r2_score(y_test_target, Y)

print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)
print("rmse:", np.sqrt(mse))
print("Model Score (R²):", lr.score(scaled_data_final_X_test, y_test_target))

