import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image

# Function to load images and labels from the folder structure
def load_image_dataset(folder_path):
    images = []
    labels = []
    for breed_folder in os.listdir(folder_path):
        breed_path = os.path.join(folder_path, breed_folder)
        if os.path.isdir(breed_path):
            for image_file in os.listdir(breed_path):
                image_path = os.path.join(breed_path, image_file)
                try:
                    # Open the image and convert to grayscale
                    with Image.open(image_path).convert('L') as img:
                        # Resize the image to a fixed size (e.g., 64x64)
                        img_resized = img.resize((64, 64))
                        # Convert image to numpy array and flatten
                        img_array = np.array(img_resized).flatten()
                        images.append(img_array)
                        labels.append(breed_folder)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return np.array(images), np.array(labels)

# Load the dataset
folder_path = "dog_breeds"
print("Loading dataset...")
X, y = load_image_dataset(folder_path)
print(f"Dataset loaded. Number of samples: {len(X)}")

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Labels encoded.")

# Split the data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("Data split.")

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized.")

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
}

# Create the MLPClassifier
print("Creating MLPClassifier...")
mlp = MLPClassifier(max_iter=1000, random_state=42)

# Perform GridSearchCV with 5-fold cross-validation
print("Performing GridSearchCV...")
# cv=5 in GridSearchCV specifies 5-fold cross-validation
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)
print("GridSearchCV completed.")

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model with the best hyperparameters
print("Training final model...")
final_model = MLPClassifier(**best_params, max_iter=1000, random_state=42)
final_model.fit(X_train_scaled, y_train)
print("Final model trained.")

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred = final_model.predict(X_test_scaled)

# Calculate precision, recall, and F1 score
print("Calculating precision, recall, and F1 score...")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate and print the confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)