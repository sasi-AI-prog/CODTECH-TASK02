import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Display the first few rows of the dataset
data.head()
from sklearn.preprocessing import StandardScaler

# Standardize numerical features
scaler = StandardScaler() #strandralize numerize the the numerical values
data_scaled = scaler.fit_transform(data.iloc[:, :-1]) #computes the mean and standard deviation for scaling and applies the transformation.
# Convert back to DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1]) #converts the scaled data back to a DataFrame format,
# Combine scaled features with target
data_scaled['target'] = data['target']
# Display the first few rows of the scaled dataset
data_scaled.head()
# Select relevant features (for demonstration purposes, we'll keep all features)
X = data_scaled.drop(columns=['target'])
y = data_scaled['target']
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #20% data used for testing and remaining used for training
# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
#25% data used for validation
# Print the shapes of the datasets
print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

Training set: (67, 4)
Validation set: (23, 4)
Test set: (30, 4)
![Screenshot 2024-07-04 201857](https://github.com/sasi-AI-prog/CODTECH-TASK02/assets/174678147/85cfc541-a97b-4803-a003-7474e694a664)
![Screenshot 2024-07-04 201923](https://github.com/sasi-AI-prog/CODTECH-TASK02/assets/174678147/b3b0af1c-d720-44af-a82f-7c72ded74180)
![Screenshot 2024-07-04 201942](https://github.com/sasi-AI-prog/CODTECH-TASK02/assets/174678147/63911fe6-1224-4367-b198-a1ee909d6543)


