import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Ensure data has the correct number of features (42)
print(f"Feature shape before training: {data.shape}")  # Debugging

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Test accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f'Accuracy: {score * 100:.2f}%')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
