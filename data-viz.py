import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Replace 'path_to_your_file.csv' with the path to your CSV file
data = pd.read_csv('C:\\Users\\Steven Cochrane\\Desktop\\data-viz\\bmiData.csv')
#print(data.head())
#print(data.describe())

# Example: Scatter plot for BMI vs. Weight
#sns.scatterplot(x='FAF', y='BMI', hue='SMOKE', data=data)
#plt.show()

# Correlation
#print(data.corr())

'''
# Mutual Information
from sklearn.feature_selection import mutual_info_regression
X = data.drop('BMI', axis=1)
y = data['BMI']
mi_scores = mutual_info_regression(X, y)
print(mi_scores)
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Prepare your data
X = data[['Weight', 'Height']]  # Example features
y = data['BMI']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print(predictions)
