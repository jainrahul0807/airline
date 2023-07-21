import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

# Load the csv file
df = pd.read_csv("Invistico_Airline.csv")
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(), inplace=True)
df['Total Delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
inflight_features = ['Seat comfort', 'Inflight wifi service', 'Inflight entertainment', 'Online support','Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling','Checkin service', 'Cleanliness', 'Online boarding']
df['Inflight Service Score'] = df[inflight_features].mean(axis=1)
df.drop(columns=['Departure Delay in Minutes', 'Arrival Delay in Minutes','Flight Distance'], inplace=True)
cols = ['Gender', 'Type of Travel', 'Class']

preprocessor= ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cols)], remainder='passthrough')

# Split the data into features (X) and target (y)
X = df.drop(columns=['satisfaction', 'Customer Type'])
y = df['satisfaction']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
rf_model.fit(X_train, y_train)


# Make pickle file of our model
pickle.dump(rf_model, open("model.pkl", "wb"))

import os
os.getcwd()

