import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Load dataset
PATH = "Combined_Crop_Recommendation.csv"
df = pd.read_csv(PATH)

# Feature selection
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Encode target labels
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target_encoded, test_size=0.2, random_state=4)

# Normalize data
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

# Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

random_search = RandomizedSearchCV(SVC(random_state=4), 
                                   param_distributions=param_grid, 
                                   n_iter=20, 
                                   cv=3, 
                                   scoring='f1_weighted', 
                                   n_jobs=-1, 
                                   random_state=4)
random_search.fit(Xtrain_scaled, Ytrain)

# Best model
svm_model = random_search.best_estimator_
svm_model.fit(Xtrain_scaled, Ytrain)

# Save model and preprocessing tools
joblib.dump(svm_model, "crop_recommendation_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Predictions
predicted_values = svm_model.predict(Xtest_scaled)

# Accuracy and F1-score
training_accuracy = svm_model.score(Xtrain_scaled, Ytrain)
testing_accuracy = svm_model.score(Xtest_scaled, Ytest)
f1_score = metrics.f1_score(Ytest, predicted_values, average='weighted')

print(f"Training Accuracy: {training_accuracy:.2f}")
print(f"Testing Accuracy: {testing_accuracy:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Confusion Matrix
conf_matrix = metrics.confusion_matrix(Ytest, predicted_values)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

print("Model and preprocessing tools saved successfully!")
