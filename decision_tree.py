import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint

# Function to load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for breed_folder in os.listdir(folder):
        breed_path = os.path.join(folder, breed_folder)
        if os.path.isdir(breed_path):
            for image_file in os.listdir(breed_path):
                image_path = os.path.join(breed_path, image_file)
                try:
                    img = Image.open(image_path).convert('RGB')
                    img = img.resize((64, 64))  # Resize images to a consistent size
                    img_array = np.array(img).flatten()  # Flatten the image into a 1D array
                    images.append(img_array)
                    labels.append(breed_folder)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return np.array(images), np.array(labels)

# Load images and labels
current_dir = os.path.dirname(__file__)
folder_path = os.path.join(current_dir, "dog_breeds")
print(f"Loading images from: {folder_path}")
X, y = load_images_from_folder(folder_path)
print(f"Loaded {len(X)} images.")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
print(f"Encoded labels: {np.unique(y)}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Define hyperparameter distributions
param_dist = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 21),
    'min_samples_leaf': randint(1, 11)
}

# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Initialize RandomizedSearchCV with parallel processing
random_search = RandomizedSearchCV(estimator=dt, param_distributions=param_dist, n_iter=20, cv=5, scoring='f1_macro', n_jobs=-1, random_state=42)

# Perform randomized search with early stopping
best_score = 0
best_params = {}
no_improvement_count = 0
max_no_improvement = 5  # Early stopping threshold

for i in range(random_search.n_iter):
    random_search.fit(X_train, y_train)
    current_best_score = random_search.best_score_
    current_best_params = random_search.best_params_
    
    if current_best_score > best_score:
        best_score = current_best_score
        best_params = current_best_params
        no_improvement_count = 0  # Reset counter if improvement is found
    else:
        no_improvement_count += 1
    
    print(f"Iteration {i+1}/{random_search.n_iter}: Best score so far: {best_score}")
    
    # Check for early stopping
    if no_improvement_count >= max_no_improvement:
        print("Early stopping triggered.")
        break

print("Best parameters:", best_params)
print("Best cross-validation score:", best_score)

# Train final model with best parameters
final_model = DecisionTreeClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
print("Final model trained.")

# Make predictions on the test set
y_pred = final_model.predict(X_test)
print("Predictions made on the test set.")

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Generate and print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)