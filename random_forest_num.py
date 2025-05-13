import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import randint
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load and preprocess image
def load_image(image_path):
    with Image.open(image_path).convert('L') as img:
        img_resized = img.resize((64, 64))  # Resize to a consistent size
        return np.array(img_resized).flatten()

# Load the dataset
data = pd.read_csv('boneage/boneage_data.csv')
logger.info("Dataset loaded")

# Load image data
image_features = []
valid_indices = []
for index, image_id in enumerate(data['id']):
    image_path = f'boneage/{image_id}.png' 
    if os.path.exists(image_path):
        image_features.append(load_image(image_path))
        valid_indices.append(index)
    if index % 100 == 0:
        logger.info(f"Processed {index} images")

# Convert image features to numpy array
image_features = np.array(image_features)
logger.info("Image features loaded")

# Filter data to include only rows with available images
data = data.iloc[valid_indices]

# Combine image features with gender
X = np.column_stack((data['male'], image_features))
y = data['boneage']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Data split into training and testing sets")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.info("Features scaled")

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 5),
    'min_samples_leaf': randint(1, 3)
}

# Create the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Perform RandomizedSearchCV with reduced iterations and cross-validation folds
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
logger.info("Starting RandomizedSearchCV")
random_search.fit(X_train_scaled, y_train)
logger.info("RandomizedSearchCV completed")

# Get the best hyperparameters
best_params = random_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train_scaled, y_train)
logger.info("Final model trained")

# Make predictions on the test set
y_pred = final_model.predict(X_test_scaled)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Generate scatter plot of true vs predicted values
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


# Residuals vs Predicted Bone Age (with gender distinction)
plt.figure(figsize=(12, 6))
male_mask = data.loc[y_test.index, 'male'] == 1
plt.scatter(y_pred[male_mask], residuals[male_mask], alpha=0.5, label='Male', color='blue')
plt.scatter(y_pred[~male_mask], residuals[~male_mask], alpha=0.5, label='Female', color='red')
plt.xlabel('Predicted Bone Age')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Bone Age (by Gender)')
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Mean Absolute Error by Age Group
age_groups = pd.cut(y_test, bins=range(0, int(y_test.max())+20, 20))
mae_by_group = pd.DataFrame({'age_group': age_groups, 'abs_error': np.abs(residuals)})
mae_by_group = mae_by_group.groupby('age_group')['abs_error'].mean()

plt.figure(figsize=(12, 6))
mae_by_group.plot(kind='bar')
plt.xlabel('Age Group')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error by Age Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()