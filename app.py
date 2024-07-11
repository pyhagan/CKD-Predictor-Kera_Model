import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
sns.set()

app = Flask(__name__)

# Load the dataset
ckd = pd.read_csv('kidney_disease_complete_2c.csv')

# Fixing data type errors and converting integer values to float
for col in ckd.select_dtypes(exclude=["object"]).columns:
    ckd[col] = ckd[col].apply(lambda x: float(x))

# Feature engineering
x = ckd.drop(["classification"], axis=1)
y = ckd["classification"]

# Balancing the labels
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(x, y)

# Normalizing data
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(X_ros)
y = y_ros

# PCA
from sklearn.decomposition import PCA
pca = PCA(0.95)
X_PCA = pca.fit_transform(x)

# Split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.2, random_state=7)

# Define the model
import keras
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout

def gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','precision'])
    return model

model = gru_model(input_shape=(x_train.shape[1], 1))
model.summary()

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=15, verbose=1)

# Evaluate the model
eval_results = model.evaluate(x_test, y_test)
loss = eval_results[0]
accuracy = eval_results[1]

# Define user input preprocessing
def preprocess_input(user_data):
    user_df = pd.DataFrame(user_data, index=[0])
    user_scaled = scaler.transform(user_df)
    user_pca = pca.transform(user_scaled)
    return user_pca

def interpret(prediction, threshold=0.5):
    if prediction >= threshold:
        return f"The patient could be suffering from Chronic Kidney Disease with a prediction probability of {prediction*100:.2f}% "
    else:
        return "The patient has a healthy kidney with a prediction probability of {:.2f}%".format((1 - prediction) * 100)

def diet_plan(result):
    if result == "The patient is likely suffering from CKD":
        diet_suggestion = """
        To improve your condition, do the following:
        
        - Consult with a dietitian for a personalized plan.
        - Drink adequate water, but not excessively.
        - Minimize intake of fizzy drinks.
        - Avoid alcohol
        - Limit protein intake to reduce the workload on kidneys.
        - Choose high-quality protein sources like fish, poultry, and eggs.
        - Reduce sodium intake to control blood pressure.
        - Limit foods high in phosphorus such as dairy, nuts, seeds.
        - Limit potassium-rich foods such as bananas, oranges, potatoes.
        """
    else:
        diet_suggestion = """
        Diet Suggestions for Healthy Kidneys:
        
        - Eat plenty of fruits and vegetables.
        - Minimize intake of fizzy drinks.
        - Minimize alcohol intake
        - Maintain a balanced diet with a variety of foods.
        - Ensure adequate hydration by drinking enough water.
        - Include whole grains and lean proteins.
        - Limit intake of processed foods and high-sodium snacks.
        - Avoid excessive amounts of sugar and saturated fats.
        """
    return diet_suggestion

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    user_data = request.json
    user_input_processed = preprocess_input(user_data)
    prediction = model.predict(user_input_processed)[0][0]
    result = interpret(prediction)
    diet_suggestion = diet_plan(result)
    
    response = {
        'prediction': result,
        'diet_suggestion': diet_suggestion
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
