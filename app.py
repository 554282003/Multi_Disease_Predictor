from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model
import secrets

app = Flask(__name__)
# Generate a secure random secret key
app.secret_key = secrets.token_hex(16)  # 32-character random string

def predict(values):
    try:
        model_map = {
            8: "models/diabetes.pkl",
            26: "models/breast_cancer.pkl",
            13: "models/heart.pkl",
            18: "models/kidney.pkl",
            10: "models/liver.pkl",
        }
        if len(values) in model_map:
            with open(model_map[len(values)], "rb") as model_file:
                model = pickle.load(model_file)
            values = np.asarray(values, dtype=np.float64).reshape(1, -1)
            return model.predict(values)[0]
        return None
    except Exception as e:
        return str(e)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/diabetes", methods=["GET", "POST"])
def diabetesPage():
    return render_template("diabetes.html")


@app.route("/cancer", methods=["GET", "POST"])
def cancerPage():
    return render_template("breast_cancer.html")


@app.route("/heart", methods=["GET", "POST"])
def heartPage():
    return render_template("heart.html")


@app.route("/kidney", methods=["GET", "POST"])
def kidneyPage():
    return render_template("kidney.html")


@app.route("/liver", methods=["GET", "POST"])
def liverPage():
    return render_template("liver.html")


@app.route("/malaria", methods=["GET", "POST"])
def malariaPage():
    return render_template("malaria.html")


@app.route("/pneumonia", methods=["GET", "POST"])
def pneumoniaPage():
    return render_template("pneumonia.html")


@app.route("/predict", methods=["POST", "GET"])
def predictPage():
    if request.method == "POST":
        try:
            to_predict_list = list(map(float, request.form.values()))
            pred = predict(to_predict_list)
            if pred is None:
                flash("Invalid input data.", "error")
                return redirect("/")
        except ValueError:
            flash("Please enter valid numerical data.", "error")
            return redirect("/")
    return render_template("predict.html", pred=pred)


@app.route("/malariapredict", methods=["POST", "GET"])
def malariapredictPage():
    if request.method == "POST":
        try:
            if "image" in request.files:
                img = Image.open(request.files["image"]).resize((36, 36))
                img = np.asarray(img).reshape((1, 36, 36, 3)).astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
            else:
                flash("Please upload an image.", "error")
                return redirect("/malaria")
        except Exception as e:
            flash(f"Error processing image: {e}", "error")
            return redirect("/malaria")
    return render_template("malaria_predict.html", pred=pred)


@app.route("/pneumoniapredict", methods=["POST", "GET"])
def pneumoniapredictPage():
    if request.method == "POST":
        try:
            if "image" in request.files:
                img = (
                    Image.open(request.files["image"])
                    .convert("L")
                    .resize((36, 36))
                )
                img = np.asarray(img).reshape((1, 36, 36, 1)) / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
            else:
                flash("Please upload an image.", "error")
                return redirect("/pneumonia")
        except Exception as e:
            flash(f"Error processing image: {e}", "error")
            return redirect("/pneumonia")
    return render_template("pneumonia_predict.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True)
