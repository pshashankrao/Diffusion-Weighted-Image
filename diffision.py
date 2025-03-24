import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.manifold import TSNE
import seaborn as sns

# Path to dataset
normal_dir = "./diffusion MRI/Normal"
ischemic_stroke_dir = "./diffusion MRI/ischemic stroke"

# Function to load and preprocess images
def load_images_from_folder(folder_path, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                img_resized = cv2.resize(img, target_size)  # Resize to target size
                images.append(img_resized)
    return np.array(images)

# Display a few images
def display_images(images, title, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"{title} {i+1}")
        plt.axis('off')
    plt.show()

# Load images from directories
normal_images = load_images_from_folder(normal_dir, target_size=(128, 128))
ischemic_images = load_images_from_folder(ischemic_stroke_dir, target_size=(128, 128))

# Display sample images from both classes
display_images(normal_images, "Normal", num_images=5)
display_images(ischemic_images, "Ischemic Stroke", num_images=5)

# Normalize images to range [0, 1]
normal_images_normalized = normal_images / 255.0
ischemic_images_normalized = ischemic_images / 255.0

# Feature extraction functions
def mean_pixel_intensity(images):
    return np.array([np.mean(img) for img in images])

def sobel_gradients(images):
    gradients = []
    for img in images:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradients.append(np.mean(magnitude))
    return np.array(gradients)

def fractal_dimension(images):
    dims = []
    for img in images:
        img = cv2.resize(img, (64, 64))  # Downsample to avoid computational overhead
        sizes = np.linspace(1, 64, 32).astype(int)
        counts = [np.sum(cv2.boxFilter(img, ddepth=-1, ksize=(s, s)) > 0) for s in sizes]
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        try:
            log_sizes = np.log(sizes)
            log_counts = np.log(counts)
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            dims.append(-slope)
        except:
            dims.append(0)  # Handle errors gracefully
    return np.array(dims)

# Extract features for both classes
normal_features = np.column_stack([mean_pixel_intensity(normal_images_normalized),
                                   sobel_gradients(normal_images_normalized),
                                   fractal_dimension(normal_images_normalized)])

ischemic_features = np.column_stack([mean_pixel_intensity(ischemic_images_normalized),
                                     sobel_gradients(ischemic_images_normalized),
                                     fractal_dimension(ischemic_images_normalized)])

# Combine features and labels
X = np.vstack([normal_features, ischemic_features])
y = np.array([0] * len(normal_features) + [1] * len(ischemic_features))

# Remove rows with NaN or inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

# Train and evaluate models
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Ischemic Stroke"], yticklabels=["Normal", "Ischemic Stroke"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc = roc_auc_score(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name} - Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.title(f"{name} - Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.show()

    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Cross-validation Score")
    plt.title(f"{name} - Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

# Plot histogram of mean pixel intensities for both classes
plt.hist(normal_features[:, 0], bins=20, alpha=0.7, label='Normal')
plt.hist(ischemic_features[:, 0], bins=20, alpha=0.7, label='Ischemic Stroke')
plt.title("Histogram of Mean Pixel Intensities")
plt.xlabel("Mean Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Box Plot for feature distributions
plt.figure(figsize=(10, 6))
plt.boxplot([normal_features[:, 0], ischemic_features[:, 0]], labels=["Normal", "Ischemic Stroke"])
plt.title("Box Plot of Mean Pixel Intensity")
plt.ylabel("Mean Intensity")
plt.show()

# Model Comparison Bar Chart
model_names = list(models.keys())
accuracies = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, color="skyblue")
plt.xlabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# T-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="coolwarm", alpha=0.7, s=100)
plt.title("T-SNE Visualization")
plt.show()

# Correlation Heatmap
correlation_matrix = np.corrcoef(X.T)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=["Mean Intensity", "Sobel Gradient", "Fractal Dimension"], yticklabels=["Mean Intensity", "Sobel Gradient", "Fractal Dimension"])
plt.title("Correlation Heatmap of Features")
plt.show()
