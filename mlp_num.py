import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from PIL import Image
import matplotlib.pyplot as plt
import os

# Function to load and preprocess image
def load_image(image_path):
    with Image.open(image_path).convert('L') as img:
        img_resized = img.resize((64, 64))  # Resize to a consistent size
        return np.array(img_resized).flatten()

# Load the dataset
data = pd.read_csv('boneage/boneage_data.csv')

# Load image data
image_features = []
valid_indices = []
for index, image_id in enumerate(data['id']):
    image_path = f'boneage/{image_id}.png'  
    if os.path.exists(image_path):
        image_features.append(load_image(image_path))
        valid_indices.append(index)

# Convert image features to numpy array
image_features = np.array(image_features)

# Filter data to include only rows with available images
data = data.iloc[valid_indices]

# Combine image features with gender
X = np.column_stack((data['male'], image_features))
y = data['boneage']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for MLP regression
mlp_regression = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(max_iter=500, early_stopping=True, n_iter_no_change=10))  # Increase iterations and add early stopping
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (100,)],
    'mlp__activation': ['relu'],
    'mlp__alpha': [0.0001, 0.001],
    'mlp__learning_rate': ['constant', 'adaptive']  # Add adaptive learning rate
}

# Perform GridSearchCV with 3-fold cross-validation and increased verbosity
grid_search = GridSearchCV(mlp_regression, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = final_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Bone Age')
plt.ylabel('Predicted Bone Age')
plt.title('True vs Predicted Bone Age')
plt.tight_layout()
plt.show()

# Calculate and plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Bone Age')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Bone Age')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()