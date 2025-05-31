import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')

### Step 1: Load & Preprocess Data
data = pd.read_csv('f2.csv')

data.rename(columns={
    'Humidity': 'Humidity', 'Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 
    'Fertilizer Name': 'Fertilizer', 'Moisture': 'Moisture', 'Nitrogen': 'Nitrogen',
    'Phosphorus': 'Phosphorus', 'Potassium': 'Potassium', 'Temperature': 'Temperature'
}, inplace=True)

data.dropna(inplace=True)

# Encoding categorical variables
encode_soil = LabelEncoder()
encode_crop = LabelEncoder()
encode_ferti = LabelEncoder()

data['Soil_Type'] = encode_soil.fit_transform(data['Soil_Type'])
data['Crop_Type'] = encode_crop.fit_transform(data['Crop_Type'])
data['Fertilizer'] = encode_ferti.fit_transform(data['Fertilizer'])

# Splitting data into train and test sets
X = data.drop('Fertilizer', axis=1)
y = data['Fertilizer']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Data normalization
scaler = MinMaxScaler().fit(x_train)
X_train_norm = scaler.transform(x_train)
X_test_norm = scaler.transform(x_test)

### Step 2: Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train_norm, y_train)

# Save model and preprocessing tools
joblib.dump(rf, "fertilizer_recommendation_model.pkl")
joblib.dump(scaler, "fertilizer_scaler.pkl")
joblib.dump(encode_soil, "soil_encoder.pkl")
joblib.dump(encode_crop, "crop_encoder.pkl")
joblib.dump(encode_ferti, "fertilizer_encoder.pkl")

# Predictions
y_pred = rf.predict(X_test_norm)

# Accuracy Scores
train_accuracy = metrics.accuracy_score(y_train, rf.predict(X_train_norm))
test_accuracy = metrics.accuracy_score(y_test, y_pred)

# F1 Score
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print(f"✅ Training Accuracy: {train_accuracy:.2f}")
print(f"✅ Test Accuracy: {test_accuracy:.2f}")
print(f"✅ F1 Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

### Step 3: ROC Curve (Multi-class)
# Binarize the labels for ROC
classes = np.unique(y)
y_test_binarized = label_binarize(y_test, classes=classes)
y_score = rf.predict_proba(X_test_norm)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap("tab10", len(classes))

for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})', color=colors(i))

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multi-class')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print("Model and preprocessing tools saved successfully!")
