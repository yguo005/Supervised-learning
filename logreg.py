import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.over_sampling import SMOTE

# Function to load images and labels from subfolders
def load_images_from_folder(folder_path, breed_to_label, image_size=(64, 64)):
    images = []
    labels = []
    for breed_name in os.listdir(folder_path):  # Iterate through each subfolder
        breed_folder = os.path.join(folder_path, breed_name)
        if os.path.isdir(breed_folder):  # Check if it's a directory
            for filename in os.listdir(breed_folder):  # Iterate through each file in the subfolder
                img_path = os.path.join(breed_folder, filename)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize(image_size)  # Resize image
                    img = np.array(img).flatten()  # Flatten the image
                    label = breed_to_label[breed_name]  # Map breed name to integer label
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Define the mapping from breed names to integer labels
breed_to_label = {
    'Poodle': 0,
    'Labrador_Retriever': 1,
    'Boxer': 2,
    'Golden_Retriever': 3,
    'Dachshund': 4,
    'Beagle': 5,
    'Rottweiler': 6,
    'Yorkshire_Terrier': 7,
    'Bulldog': 8,
    'German_Shepherd': 9
}

# Load images and labels
folder_path = '/Users/mac/Desktop/5100/Assignment2/dog_breeds'
X, y = load_images_from_folder(folder_path, breed_to_label)

# Check if any images were loaded
if len(X) == 0 or len(y) == 0:
    raise ValueError("No images were loaded. Please check the folder path and image files.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define logistic regression model with hyperparameter tuning and class weights
param_grid = {'C': [0.1, 1, 10, 100]}
logreg = LogisticRegression(class_weight='balanced', max_iter=1000)  # Increased max_iter

# Use StratifiedKFold with 3 splits
cv = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(logreg, param_grid, cv=cv, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# Retrieve and print the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Evaluate model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Precision:", precision_score(y_test, y_pred, average='macro', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='macro', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, average='macro', zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))