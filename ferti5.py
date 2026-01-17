import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Configuration ---
MODEL_FILE = 'fertilizer_model.pkl'
ENCODER_FILE = 'label_encoder.pkl'
CATEGORIES_FILE = 'categories.pkl'

# --- 1. Load Data ---
try:
    df = pd.read_csv("D:\\soilproject\\dataset\\data_core.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data_core.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Rename columns for simpler access
df.columns = df.columns.str.strip().str.replace(' ', '_')

# --- 2. Data Preprocessing and Feature Engineering ---
X = df.drop('Fertilizer_Name', axis=1)
y_raw = df['Fertilizer_Name']

# 2a. Encode the target variable (Fertilizer Name)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
target_names = label_encoder.classes_

print(f"\nUnique Fertilizer Types: {target_names}")
print(f"Total samples: {len(df)}")

# 2b. Define features
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil_Type', 'Crop_Type']

# Extract unique categories for Flask App
soil_types = df['Soil_Type'].unique().tolist()
crop_types = df['Crop_Type'].unique().tolist()
print(f"Unique Soil Types: {soil_types}")
print(f"Unique Crop Types: {crop_types}")

# 2c. Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")


# --- 4. Model Selection and Hyperparameter Tuning ---
best_model = None
best_accuracy = 0.0
best_model_name = ""

# Define models and their hyperparameter grids
models_to_tune = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {
            # Reducing estimators and depth significantly shrinks the .pkl file size
            'classifier__n_estimators': [50, 80], 
            'classifier__max_depth': [10, 15],
            'classifier__min_samples_leaf': [2, 4], 
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.1],
            'classifier__max_depth': [3, 5],
        }
    }
}

print("\nStarting Hyperparameter Tuning and Model Comparison...")

for name, config in models_to_tune.items():
    print(f"\n--- Tuning {name} ---")

    # Create a pipeline specific to the current model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])

    # Use GridSearchCV for hyperparameter optimization
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config['param_grid'],
        cv=5, # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Evaluate the best model found by the grid search on the test set
    tuned_pipeline = grid_search.best_estimator_
    y_pred = tuned_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} Test Accuracy: {accuracy:.4f}")

    # Check if this is the best model so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = tuned_pipeline
        best_model_name = name
        print(f"*** {name} is the new best model! ***")


# --- 5. Final Evaluation and Conclusion ---
print("\n--- Model Selection Complete ---")
print(f"The best performing model is: {best_model_name} with an Accuracy of {best_accuracy:.4f}")

if best_accuracy > 0.90:
    print("\n✅ Success! Target accuracy of >90% achieved!")
else:
    print(f"\n⚠️ Target accuracy of >90% not met. Current best is {best_accuracy:.4f}. More data or feature engineering may be required.")

# Print detailed report for the best model
y_pred_final = best_model.predict(X_test)
print("\nClassification Report for the Best Model:")
print(classification_report(y_test, y_pred_final, target_names=target_names))


# --- 6. Save Best Model and Encoder ---
# Using compress=3 reduces file size by another 20-40%
joblib.dump(best_model, MODEL_FILE, compress=3) 
joblib.dump(label_encoder, ENCODER_FILE)

# Save categories for the Flask App
categories_data = {
    'soil_types': soil_types,
    'crop_types': crop_types
}
joblib.dump(categories_data, CATEGORIES_FILE)

print(f"\nBest Model ({best_model_name}) saved to: {MODEL_FILE}")
print(f"Label Encoder saved to: {ENCODER_FILE}")
print(f"Categories saved to: {CATEGORIES_FILE}")
