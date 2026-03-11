import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('water_potability.csv')

# 1. Clean Data: Fill missing values with the median
imputer = SimpleImputer(strategy='median')
df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 2. Setup Features and Target
X = df_clean.drop('Potability', axis=1)
y = df_clean['Potability']

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# 1. Distribution of the Target (Pie Chart)
# Shows the imbalance between Potable and Non-Potable samples
plt.figure(figsize=(7,7))
df['Potability'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], labels=['Non-Potable', 'Potable'])
plt.title('Distribution of Water Potability (SDG 6 Data)')
plt.show()

# 2. Correlation Heatmap
# Helps identify if any chemicals are strongly linked together
plt.figure(figsize=(12, 10))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Water Quality Features')
plt.show()

# 3. Feature Importance (Why the model chose what it did)
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='teal')
plt.title('Most Critical Water Factors (Random Forest Insights)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# 4. Confusion Matrix (Visualizing Errors)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-Potable', 'Potable'], yticklabels=['Non-Potable', 'Potable'])
plt.xlabel('Model Prediction')
plt.ylabel('Actual Truth')
plt.title('Confusion Matrix: Accuracy Breakdown')
plt.show()

# 5. pH Distribution vs Potability (Scientific Insight)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_clean, x='ph', hue='Potability', fill=True, palette='magma')
plt.title('pH Levels: Potable vs Non-Potable Water')
plt.axvline(x=6.5, color='red', linestyle='--', label='WHO Min Safe')
plt.axvline(x=8.5, color='red', linestyle='--', label='WHO Max Safe')
plt.legend()
plt.show()