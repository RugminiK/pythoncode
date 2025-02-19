#Q1. 
import pandas as pd
import numpy as np

file_path = r"C:\Users\vero\Downloads\Lab Session Data.xlsx"

df = pd.read_excel(file_path, sheet_name="Purchase data")

# Display available columns to verify structure
print("Columns in dataset:", df.columns)

# Drop the "Customer" column (non-numeric)
df = df.drop(columns=["Customer"])

df = df.apply(pd.to_numeric, errors="coerce")

df.fillna(0, inplace=True)  

A = df.iloc[:, :-1].values  # All columns except last one
C = df.iloc[:, -1].values   # Last column (dependent variable)

# Compute vector space properties
dimensionality = A.shape[1]
num_vectors = A.shape[0]
rank_A = np.linalg.matrix_rank(A) 

# Compute pseudo-inverse of A
A_pinv = np.linalg.pinv(A)

# Solve for X (cost of each product)
X = np.dot(A_pinv, C)

# Display results
print(f"Dimensionality of vector space: {dimensionality}")
print(f"Number of vectors: {num_vectors}")
print(f"Rank of Matrix A: {rank_A}")
print(f"Cost of each product:\n{X}")

#Q2.
import numpy as np

# Compute model vector X using pseudo-inverse
X_model = np.linalg.pinv(A) @ C

# Print result
print(f"Model Vector X: {X_model.flatten()}")

#Q3.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
df_purchase['Category'] = df_purchase['Payments'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Prepare data for classification
X = df_purchase.drop(columns=['Category'])
y = df_purchase['Category']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)  # Convert RICH/POOR to 1/0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Classifier Accuracy: {accuracy}")

#Q4.
import pandas as pd
import statistics
import seaborn as sns
import matplotlib.pyplot as plt

df_stock = pd.read_excel(data_file, sheet_name="IRCTC Stock Price")

# Calculate mean and variance
mean_price = statistics.mean(df_stock['Price'])
variance_price = statistics.variance(df_stock['Price'])

# Extract Wednesday price data
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
wednesday_prices = df_stock[df_stock['Date'].dt.day_name() == 'Wednesday']['Price']
sample_mean_wed = statistics.mean(wednesday_prices)

# Compute probabilities
loss_prob = sum(df_stock['Chg%'] < 0) / len(df_stock)
profit_wed_prob = sum((df_stock['Chg%'] > 0) & (df_stock['Date'].dt.day_name() == 'Wednesday')) / sum(df_stock['Date'].dt.day_name() == 'Wednesday')

# Print results
print(f"Mean Price: {mean_price}, Variance: {variance_price}")
print(f"Wednesday Sample Mean: {sample_mean_wed}")
print(f"Probability of Loss: {loss_prob}")
print(f"Probability of Profit on Wednesday: {profit_wed_prob}")

# Scatter plot of Change % vs. Day of the Week
df_stock['Day'] = df_stock['Date'].dt.day_name()
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_stock['Day'], y=df_stock['Chg%'])
plt.xlabel("Day of the Week")
plt.ylabel("Change %")
plt.title("Stock Price Change % vs Day of the Week")
plt.show()

#Q5.
import pandas as pd
import numpy as np

# Load the dataset
file_path = "Lab Session Data (1).xlsx"  
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Identify data types
print(df.dtypes)

# Checking missing values
print("Missing values per column:\n", df.isnull().sum())

# Checking for outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Outliers per column:\n", outliers)

# Mean and variance of numerical variables
print("Mean:\n", df.mean())
print("Variance:\n", df.var())

# Encoding categorical attributes
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
#Q6.
# Filling missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'float64':  
            df[col].fillna(df[col].median(), inplace=True)  # Median for numeric with outliers
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)  # Mode for categorical

print("Missing values after imputation:\n", df.isnull().sum())

#Q7.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Data after normalization:\n", df.head())

#Q8.
# Converting first two rows into binary vectors
vector1 = df.iloc[0].astype(bool)
vector2 = df.iloc[1].astype(bool)

f11 = np.sum(vector1 & vector2)
f00 = np.sum(~vector1 & ~vector2)
f01 = np.sum(~vector1 & vector2)
f10 = np.sum(vector1 & ~vector2)

JC = f11 / (f01 + f10 + f11)
SMC = (f11 + f00) / (f00 + f01 + f10 + f11)

print("Jaccard Coefficient:", JC)
print("Simple Matching Coefficient:", SMC)

#Q9.
from scipy.spatial.distance import cosine

vec1 = df.iloc[0].values
vec2 = df.iloc[1].values
cos_sim = 1 - cosine(vec1, vec2)

print("Cosine Similarity:", cos_sim)

#Q10.
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Compute cosine similarity for the first 20 vectors
similarity_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        similarity_matrix[i, j] = 1 - cosine(df.iloc[i].values, df.iloc[j].values)

# Heatmap plot
sns.heatmap(similarity_matrix, annot=True)
plt.title("Heatmap of Cosine Similarities")
plt.show()
