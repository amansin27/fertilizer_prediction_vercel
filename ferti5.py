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
    # Use a relative path or ensure the file exists in the directory for portability
    df = pd.read_csv("D:\\soilproject\\dataset\\data_core.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data_core.csv' not found.")
    exit()

df.columns = df.columns.str.strip().str.replace(' ', '_')

# --- 2. Preprocessing ---
X = df.drop('Fertilizer_Name', axis=1)
y_raw = df['Fertilizer_Name']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
target_names = label_encoder.classes_

numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil_Type', 'Crop_Type']

soil_types = df['Soil_Type'].unique().tolist()
crop_types = df['Crop_Type'].unique().tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Model Selection with Size Constraints ---
print("\nStarting Hyperparameter Tuning with Size Constraints...")

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

best_model = None
best_accuracy = 0.0
best_model_name = ""

for name, config in models_to_tune.items():
    print(f"\n--- Tuning {name} ---")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config['param_grid'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    tuned_pipeline = grid_search.best_estimator_
    accuracy = accuracy_score(y_test, tuned_pipeline.predict(X_test))

    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Test Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = tuned_pipeline
        best_model_name = name

# --- 5. Save Model (Using Compression) ---
# Using compress=3 reduces file size by another 20-40%
joblib.dump(best_model, MODEL_FILE, compress=3) 
joblib.dump(label_encoder, ENCODER_FILE)
joblib.dump({'soil_types': soil_types, 'crop_types': crop_types}, CATEGORIES_FILE)

print(f"\nSaved optimized {best_model_name} to {MODEL_FILE}")