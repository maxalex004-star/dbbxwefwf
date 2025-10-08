from flask import Flask, request, jsonify

from sklearn.linear_model import LinearRegression

import numpy as np

 

app = Flask(__name__)

 

@app.route('/predict', methods=['POST'])

def predict():

    try:

        data = request.get_json()

        months = np.array(data["months"]).reshape(-1, 1)

        values = np.array(data["values"])

        future = np.array(data["future"]).reshape(-1, 1)

 

        model = LinearRegression()

        model.fit(months, values)

        predictions = model.predict(future)

 

        return jsonify(predictions.tolist())

    except Exception as e:

        return jsonify({"error": str(e)}), 400

 

if __name__ == '__main__':

    app.run()
