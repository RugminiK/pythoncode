import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:/Users/vero/OneDrive/Documents/acads/4th_sem/ML/data.csv"
df = pd.read_csv(file_path)

# Drop non-numeric column
df = df.drop(columns=['crop'])

# Convert all columns to numeric, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values
df = df.dropna()

# A1: Compute centroids, spread, and interclass distance
class_0 = df[df['pump'] == 0].drop(columns=['pump']).values
class_1 = df[df['pump'] == 1].drop(columns=['pump']).values
centroid_0 = np.mean(class_0, axis=0)
centroid_1 = np.mean(class_1, axis=0)
spread_0 = np.std(class_0, axis=0)
spread_1 = np.std(class_1, axis=0)
distance_classes = np.linalg.norm(centroid_0 - centroid_1)
print("Centroid of Class 0:", centroid_0)
print("Centroid of Class 1:", centroid_1)
print("Spread of Class 0:", spread_0)
print("Spread of Class 1:", spread_1)
print("Interclass Distance:", distance_classes)

# A2: Histogram for moisture
plt.hist(df['moisture'], bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Moisture")
plt.ylabel("Frequency")
plt.title("Histogram of Moisture")
plt.show()
print("Mean of Moisture:", np.mean(df['moisture']))
print("Variance of Moisture:", np.var(df['moisture']))

# A3: Minkowski distance
feature_1 = df.iloc[0, 1:3].values
feature_2 = df.iloc[1, 1:3].values
minkowski_distances = [distance.minkowski(feature_1, feature_2, p=r) for r in range(1, 11)]
plt.plot(range(1, 11), minkowski_distances, marker='o')
plt.xlabel("r value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs r")
plt.grid()
plt.show()

# A4: Divide dataset into train and test set
X = df.drop(columns=['pump']).values
y = df['pump'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# A5: Train kNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("kNN Classifier trained with k=3")

# A6: Test accuracy
accuracy = knn.score(X_test, y_test)
print("kNN Classifier Accuracy:", accuracy)

# A7: Predict test set
y_pred = knn.predict(X_test)
print("Predicted classes for test set:", y_pred)

# A8: Vary k and plot accuracy
k_values = range(1, 12)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("kNN Accuracy vs k value")
plt.grid()
plt.show()

# A9: Evaluate confusion matrix and classification metrics
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
