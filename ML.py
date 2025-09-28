import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import random
# Load dataset from CSV
df = pd.read_csv("equipment_anomaly_data (1).csv")

# Apply transformations
# 1. Temperature: keep rows in [50,90], then subtract 30
df = df[(df['temperature'] >= 50) & (df['temperature'] <= 90)]
df['temperature'] = df['temperature'] - 30

# 2. Pressure: keep rows in [18,50]
df = df[(df['pressure'] >= 18) & (df['pressure'] <= 50)]

# 3. Vibration: keep rows in [1,4], then add random +10...15
df = df[(df['vibration'] >= 1) & (df['vibration'] <= 4)]
df['vibration'] = df['vibration'].apply(lambda x: x + random.randint(15, 20))

# 4. Drop humidity & location
df = df.drop(columns=['humidity', 'location'])

# Force equipment names mapping
df['equipment'] = df['equipment'].replace({
    "Compressor": "pump101",
    "Turbine": "pump102"
})

# Drop أي صفوف ما صارت pump101 أو pump102
df = df[df['equipment'].isin(["pump101", "pump102"])]

# ================= STEP 1.2: Encode Equipment =================
encoder = LabelEncoder()
df['equipment_code'] = encoder.fit_transform(df['equipment'])

# ================= STEP 1.3: Create Lag Features =================
df['last_temp'] = df.groupby('equipment')['temperature'].shift(1)
df['last_vib'] = df.groupby('equipment')['vibration'].shift(1)
df['last_pressure'] = df.groupby('equipment')['pressure'].shift(1)

# ================= STEP 1.4: Change Features =================
df['temp_change'] = df['temperature'] - df['last_temp']
df['vibration_change'] = df['vibration'] - df['last_vib']
df['pressure_change'] = df['pressure'] - df['last_pressure']

# ================= STEP 1.5: Percentage Changes =================
df['temp_change_pct'] = (df['temp_change'] / df['last_temp']) * 100
df['vibration_change_pct'] = (df['vibration_change'] / df['last_vib']) * 100
df['pressure_change_pct'] = (df['pressure_change'] / df['last_pressure']) * 100

# Fill NaN with 0
df.fillna(0, inplace=True)


# ================= STEP 1.6: Features & Labels =================
X = df[['equipment_code','temperature','vibration','pressure',
        'temp_change','vibration_change','pressure_change',
        'temp_change_pct','vibration_change_pct','pressure_change_pct']]
y = df['faulty']  # <-- from CSV

# ================= STEP 1.7: Feature Selection =================
selector = SelectKBest(score_func=f_classif, k=8)
X_selected_temp = selector.fit_transform(X.drop(columns=['equipment_code']), y)
selected_features = list(X.drop(columns=['equipment_code']).columns[selector.get_support()])
selected_features.append('equipment_code')  # always include



# Final feature matrix
X_selected = df[selected_features].values

# ================= STEP 1.8: Balance with SMOTE =================
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# ================= STEP 1.9: Train/Test Split =================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# ================= STEP 1.10: Train Decision Tree =================
model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
model.fit(X_train, y_train)

import joblib

# ✅ حفظ الموديل
joblib.dump(model, "decision_tree.pkl")

# ✅ حفظ الـ encoder
joblib.dump(encoder, "encoder.pkl")

# ✅ حفظ قائمة الميزات
joblib.dump(selected_features, "selected_features.pkl")

print("✅ Saved: decision_tree.pkl, encoder.pkl, selected_features.pkl")

# Predictions
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Decision rules
tree_rules = export_text(model, feature_names=list(selected_features))
print("\n🌳 Decision Tree Rules:\n", tree_rules)

# Step 11: Predict risk for each equipment
results = []
for i, row in df.iterrows():
    features = row[selected_features].values.reshape(1, -1)
    risk_score = model.predict_proba(features)[0][1] * 100
    status = "⚠️ Failure Risk" if risk_score > 85 else "✅ Normal"
    results.append([row['equipment'], row['temperature'], row['vibration'],
                    row['pressure'], f"{risk_score:.2f}%", status])

# Show results in table
results_df = pd.DataFrame(results, columns=['Equipment','Temp','Vibration','Pressure','Risk Score','Status'])
print("\n📊 Final Results:\n", results_df)
