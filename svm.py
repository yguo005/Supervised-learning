import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

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
folder_path = "dog_breeds"
X, y = load_images_from_folder(folder_path)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter ranges
C_range = [0.1, 1, 10, 100]
kernel_range = ['rbf', 'linear']
gamma_range = ['scale', 'auto', 0.1, 1]

# Initialize variables to store best hyperparameters and score
best_score = 0
best_params = {}

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for C in C_range:
    for kernel in kernel_range:
        for gamma in gamma_range:
            svm = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
            scores = cross_val_score(svm, X_train, y_train, cv=kf, scoring='f1_macro')
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}

print("Best parameters:", best_params)
print("Best cross-validation score:", best_score)

# Train final model with best parameters
final_model = SVC(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = final_model.predict(X_test)

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Generate and print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

