from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

model_path = "crop_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Train the model first.")

with open(model_path, "rb") as f:
    model = pickle.load(f)

crop_images = {
    "wheat": "static/images/wheat.jpg",
    "rice": "static/images/rice.jpg",
    "maize": "static/images/maize.jpg",
    "chickpea": "static/images/chickpea.jpg",
    "kidneybeans": "static/images/kidneybeans.jpg",
    "pigeonpeas": "static/images/pigeonpeas.jpg",
    "mothbeans": "static/images/mothbeans.jpg",
    "mungbean": "static/images/mungbean.jpg",
    "blackgram": "static/images/blackgram.jpg",
    "lentil": "static/images/lentil.jpg",
    "pomegranate": "static/images/pomegranate.jpg",
    "banana" : "static/images/banana.jpg",
    "mango" : "static/images/mango.jpg",
    "grapes" : "static/images/grapes.jpg",
    "watermelon" : "static/images/watermelon.jpg",
    "muskmelon" : "static/images/muskmelon.jpg",
    "apple" : "static/images/apple.jpg",
    "orange" : "static/images/orange.jpg",
    "papaya" : "static/images/papaya.jpg",
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_crop():
    try:
        data = request.get_json()

        soil_type = data.pop("Soil Type", None)
        soil_types = ["Clayey", "Loamy", "Sandy","Sandy loam"]

        for soil in soil_types:
            data[f"Soil Type_{soil}"] = 1 if soil_type == soil else 0

        df = pd.DataFrame([data])

        column_order = ["N", "P", "K", "Temperature", "humidity", "ph", "Rainfall", 
                        "Soil Type_Clayey", "Soil Type_Loamy", "Soil Type_Sandy" ,"Soil Type_Sandy loam"]
        df = df.reindex(columns=column_order, fill_value=0)

        crop = model.predict(df)[0]

        image_path = crop_images.get(crop.lower(), "static/images/default.jpg")

        return jsonify({"recommended_crop": crop, "image": image_path})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)