import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open(r"model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    gender=request.form['gender']
    age=int(request.form['age'])
    type_of_travel=request.form['type_of_travel']
    cust_class=request.form['class']
    flight_distance=int(request.form['flightDistance'])
    dep_delay=int(request.form['departureDelay'])
    arr_delay=int(request.form['arrivalDelay'])
    seat_comfort=int(request.form['seatComfort'])
    deparr_conv=int(request.form['depArrConv'])
    food_drink=int(request.form['foodDrink'])
    gate_location=int(request.form['gateLocation'])
    in_flight_wifi=int(request.form['inFlightWifi'])
    in_flight_entertainment=int(request.form['inFlightEntertainment'])
    online_support=int(request.form['onlineSupport'])
    ease_of_booking=int(request.form['easeOfBooking'])
    on_board_service=int(request.form['onBoardService'])
    legroom_service=int(request.form['legroomService'])
    baggage_handling=int(request.form['baggageHandling'])
    checkin_service=int(request.form['checkinService'])    
    cleanliness=int(request.form['cleanliness'])
    online_boarding=int(request.form['onlineBoarding'])
    tot_delay=dep_delay+arr_delay
    inflight_features = [seat_comfort, in_flight_wifi, in_flight_entertainment, online_support, ease_of_booking, on_board_service, legroom_service, baggage_handling, checkin_service, cleanliness, online_boarding]
    inflight_score = sum(inflight_features)/len(inflight_features)

    response_data=pd.DataFrame([[gender, age, type_of_travel, cust_class, flight_distance, seat_comfort, deparr_conv, food_drink, gate_location, in_flight_wifi, in_flight_entertainment, online_support, ease_of_booking, on_board_service, legroom_service, baggage_handling, checkin_service, cleanliness, online_boarding, tot_delay, inflight_score]], columns=['Gender', 'Age', 'Type of Travel', 'Class','Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient', 'Food and drink', 'Gate location', 'Inflight wifi service',
    'Inflight entertainment', 'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding', 'Total Delay', 'Inflight Service Score'])
    prediction = model.predict(response_data)
    if prediction[0] == 0:
        result = 'Not Satisfied'
    else:
        result = 'Satisfied'

    return render_template("index.html", predicted_value = "The Predicted Value is {}".format(result))

if __name__ == "__main__":
    flask_app.run(debug=True)