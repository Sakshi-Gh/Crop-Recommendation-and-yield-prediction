# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/crop_recommend.csv")

features = df[['N', 'P','K','pH', 'rainfall','temperature']]
target = df['Crop']
labels = df['Crop']

# Splitting into train and test data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Initialize the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],  # Criteria for split quality
    'max_depth': [None, 10, 20, 30],  # Depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split
    'min_samples_leaf': [1, 2, 5],    # Minimum samples required at a leaf
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=1
)

# Train the model with GridSearchCV
grid_search.fit(Xtrain, Ytrain)

# Best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Predict on the test data using the best model
predicted_values = best_model.predict(Xtest)

# Calculate accuracy
x = accuracy_score(Ytest, predicted_values)
print("Tuned Decision Tree's Accuracy is: ", x)

# Print classification report
print(classification_report(Ytest, predicted_values))

# Save the model
pickle.dump(best_model, open('model.pkl', 'wb'))

