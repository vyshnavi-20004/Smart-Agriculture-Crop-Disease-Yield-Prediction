from flask import Flask, render_template, request, send_from_directory
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from PIL import Image


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


DISEASE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "crop_disease_model.keras"
)


CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)


YIELD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yield_prediction_model.joblib"
)


# =========================================================
# UPLOAD CONFIGURATION
# =========================================================

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# LOAD DISEASE DETECTION MODEL
# =========================================================

print("Loading AI disease detection model...")


model = tf.keras.models.load_model(
    DISEASE_MODEL_PATH
)


with open(
    CLASS_NAMES_PATH,
    "r"
) as f:

    class_names = json.load(f)


print(
    "AI disease model loaded successfully!"
)


print(
    "Number of disease classes:",
    len(class_names)
)


# =========================================================
# LOAD YIELD PREDICTION MODEL
# =========================================================

print(
    "Loading AI yield prediction model..."
)


yield_model = joblib.load(
    YIELD_MODEL_PATH
)


print(
    "AI yield prediction model loaded successfully!"
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# SERVE UPLOADED IMAGES
# =========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# =========================================================
# CROP DISEASE DETECTION
# =========================================================

@app.route(
    "/disease",
    methods=["GET", "POST"]
)
def disease():

    result = None

    image_path = None

    prediction = None

    confidence = None

    status = None


    # -----------------------------------------------------
    # POST REQUEST
    # -----------------------------------------------------

    if request.method == "POST":

        image = request.files.get(
            "crop-image"
        )


        # -------------------------------------------------
        # CHECK IMAGE
        # -------------------------------------------------

        if image and image.filename != "":

            filename = image.filename


            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )


            # Save uploaded image

            image.save(
                filepath
            )


            # -------------------------------------------------
            # IMPORTANT:
            # Give browser a proper Flask URL
            # -------------------------------------------------

            image_path = (
                "/uploads/"
                + os.path.basename(filepath)
            )


            try:

                # ---------------------------------------------
                # OPEN IMAGE
                # ---------------------------------------------

                img = Image.open(
                    filepath
                ).convert("RGB")


                # ---------------------------------------------
                # RESIZE IMAGE
                # ---------------------------------------------

                img = img.resize(
                    (160, 160)
                )


                # ---------------------------------------------
                # CONVERT TO NUMPY ARRAY
                # ---------------------------------------------

                img_array = np.array(
                    img
                )


                # ---------------------------------------------
                # ADD BATCH DIMENSION
                # ---------------------------------------------

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )


                # ---------------------------------------------
                # AI PREDICTION
                # ---------------------------------------------

                predictions = model.predict(
                    img_array,
                    verbose=0
                )


                # ---------------------------------------------
                # FIND PREDICTED CLASS
                # ---------------------------------------------

                predicted_index = np.argmax(
                    predictions[0]
                )


                # ---------------------------------------------
                # CONFIDENCE
                # ---------------------------------------------

                confidence = float(
                    predictions[0][predicted_index]
                    * 100
                )


                # ---------------------------------------------
                # DISEASE NAME
                # ---------------------------------------------

                prediction = class_names[
                    predicted_index
                ]


                # ---------------------------------------------
                # HEALTHY / DISEASE STATUS
                # ---------------------------------------------

                if "healthy" in prediction.lower():

                    status = "Healthy Crop"

                else:

                    status = "Disease Detected"


                # ---------------------------------------------
                # SUCCESS MESSAGE
                # ---------------------------------------------

                result = (
                    "AI analysis completed successfully!"
                )


            except Exception as e:

                result = (
                    f"Prediction error: {str(e)}"
                )


        else:

            result = (
                "Please select a crop image."
            )


    # =====================================================
    # SEND RESULT TO disease.html
    # =====================================================

    return render_template(

        "disease.html",

        result=result,

        image_path=image_path,

        prediction=prediction,

        confidence=confidence,

        status=status

    )


# =========================================================
# CROP YIELD PREDICTION
# =========================================================

@app.route(
    "/yield-prediction",
    methods=["GET", "POST"]
)
def yield_prediction():

    prediction = None

    total_production = None

    error = None


    # -----------------------------------------------------
    # DEFAULT FORM DATA
    # -----------------------------------------------------

    form_data = {

        "crop": "",

        "crop_year": "",

        "season": "",

        "state": "",

        "area": "",

        "rainfall": "",

        "fertilizer": "",

        "pesticide": ""

    }


    # =====================================================
    # POST REQUEST
    # =====================================================

    if request.method == "POST":

        try:

            # ---------------------------------------------
            # GET FORM VALUES
            # ---------------------------------------------

            crop = request.form.get(
                "crop",
                ""
            ).strip()


            crop_year = int(
                request.form.get(
                    "crop_year"
                )
            )


            season = request.form.get(
                "season",
                ""
            ).strip()


            state = request.form.get(
                "state",
                ""
            ).strip()


            area = float(
                request.form.get(
                    "area"
                )
            )


            rainfall = float(
                request.form.get(
                    "rainfall"
                )
            )


            fertilizer = float(
                request.form.get(
                    "fertilizer"
                )
            )


            pesticide = float(
                request.form.get(
                    "pesticide"
                )
            )


            # ---------------------------------------------
            # SAVE FORM DATA
            # ---------------------------------------------

            form_data = {

                "crop": crop,

                "crop_year": crop_year,

                "season": season,

                "state": state,

                "area": area,

                "rainfall": rainfall,

                "fertilizer": fertilizer,

                "pesticide": pesticide

            }


            # ---------------------------------------------
            # CREATE DATAFRAME
            # ---------------------------------------------

            input_data = pd.DataFrame([{

                "Crop": crop,

                "Crop_Year": crop_year,

                "Season": season,

                "State": state,

                "Area": area,

                "Annual_Rainfall": rainfall,

                "Fertilizer": fertilizer,

                "Pesticide": pesticide

            }])


            # ---------------------------------------------
            # AI YIELD PREDICTION
            # ---------------------------------------------

            predicted_yield = float(
                yield_model.predict(
                    input_data
                )[0]
            )


            # ---------------------------------------------
            # PREVENT NEGATIVE PREDICTION
            # ---------------------------------------------

            predicted_yield = max(
                0,
                predicted_yield
            )


            # ---------------------------------------------
            # TOTAL PRODUCTION
            # ---------------------------------------------

            total_production = (
                predicted_yield * area
            )


            # ---------------------------------------------
            # ROUND RESULTS
            # ---------------------------------------------

            prediction = round(
                predicted_yield,
                2
            )


            total_production = round(
                total_production,
                2
            )


        except Exception as e:

            error = (
                f"Prediction error: {str(e)}"
            )


    # =====================================================
    # SEND RESULT TO yield.html
    # =====================================================

    return render_template(

        "yield.html",

        prediction=prediction,

        total_production=total_production,

        error=error,

        form_data=form_data

    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )