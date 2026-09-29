import os

# Reduce TensorFlow log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, send_from_directory
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from PIL import Image


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

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

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# TensorFlow configuration
# =========================================================

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# =========================================================
# LOAD DISEASE MODEL
# =========================================================

print("Loading AI disease detection model...")

model = tf.keras.models.load_model(
    DISEASE_MODEL_PATH,
    compile=False
)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("AI disease model loaded successfully!")
print("Number of disease classes:", len(class_names))


# =========================================================
# LOAD YIELD MODEL
# =========================================================

print("Loading AI yield prediction model...")

yield_model = joblib.load(YIELD_MODEL_PATH)

print("AI yield prediction model loaded successfully!")


# =========================================================
# LIGHTWEIGHT IMAGE VALIDATION
# =========================================================

def validate_crop_image(filepath):

    try:

        # Open image
        image = Image.open(filepath).convert("RGB")

        # Very small images are usually not useful for analysis
        width, height = image.size

        if width < 100 or height < 100:
            return False

        # Resize image for lightweight analysis
        image = image.resize((128, 128))

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        red = image_array[:, :, 0]
        green = image_array[:, :, 1]
        blue = image_array[:, :, 2]

        # =================================================
        # GREEN PIXEL CHECK
        # =================================================

        green_pixels = (
            (green > red * 1.05) &
            (green > blue * 1.02) &
            (green > 45)
        )

        green_ratio = float(
            np.mean(green_pixels)
        )

        # =================================================
        # NATURAL COLOR CHECK
        # =================================================

        # Brown/yellow/green colors are also common
        # in leaves and diseased crop images.

        plant_color_pixels = (

            # Green
            (
                (green > red * 1.05) &
                (green > blue * 1.02) &
                (green > 45)
            )

            |

            # Yellow
            (
                (red > 80) &
                (green > 70) &
                (blue < 100) &
                (red > blue * 1.2)
            )

            |

            # Brown
            (
                (red > blue * 1.25) &
                (green > blue * 1.10) &
                (red > 60) &
                (green < 180)
            )
        )

        plant_color_ratio = float(
            np.mean(plant_color_pixels)
        )

        # =================================================
        # IMAGE VARIATION CHECK
        # =================================================

        # Leaves usually contain some color variation.
        # This helps reject completely plain images.

        red_std = float(np.std(red))
        green_std = float(np.std(green))
        blue_std = float(np.std(blue))

        color_variation = (
            red_std +
            green_std +
            blue_std
        )

        # =================================================
        # FINAL VALIDATION
        # =================================================

        print(
            "Green ratio:",
            round(green_ratio * 100, 2),
            "%"
        )

        print(
            "Plant color ratio:",
            round(plant_color_ratio * 100, 2),
            "%"
        )

        print(
            "Color variation:",
            round(color_variation, 2)
        )

        # Strong green image
        if green_ratio >= 0.08:
            return True

        # Significant natural plant-like colors
        if plant_color_ratio >= 0.20 and color_variation >= 35:
            return True

        # Mostly green/yellow/brown crop image
        if plant_color_ratio >= 0.12 and color_variation >= 55:
            return True

        return False

    except Exception as e:

        print(
            "Image validation error:",
            e
        )

        return False


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOADED IMAGES
# =========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# =========================================================
# DISEASE DETECTION
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

    if request.method == "POST":

        image = request.files.get(
            "crop-image"
        )

        if image and image.filename != "":

            filename = image.filename

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(filepath)

            image_path = (
                "/uploads/" +
                os.path.basename(filepath)
            )

            try:

                # ==========================================
                # STEP 1: LIGHTWEIGHT IMAGE VALIDATION
                # ==========================================

                is_valid_crop = (
                    validate_crop_image(
                        filepath
                    )
                )

                if not is_valid_crop:

                    status = "Invalid Image"

                    result = (
                        "This image does not appear "
                        "to be a crop or leaf image. "
                        "Please upload a valid crop "
                        "leaf image."
                    )

                    return render_template(
                        "disease.html",
                        result=result,
                        image_path=image_path,
                        prediction=None,
                        confidence=None,
                        status=status
                    )

                # ==========================================
                # STEP 2: DISEASE PREDICTION
                # ==========================================

                img = Image.open(
                    filepath
                ).convert("RGB")

                img = img.resize(
                    (160, 160)
                )

                img_array = np.asarray(
                    img,
                    dtype=np.float32
                )

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                predictions = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(
                        predictions[0]
                    )
                )

                confidence = float(
                    predictions[0][
                        predicted_index
                    ] * 100
                )

                prediction = (
                    class_names[
                        predicted_index
                    ]
                )

                # ==========================================
                # CONFIDENCE CHECK
                # ==========================================

                if confidence < 60:

                    status = (
                        "Uncertain Image"
                    )

                    result = (
                        "The image could not be "
                        "classified with sufficient "
                        "confidence. Please upload "
                        "a clear crop leaf image."
                    )

                    prediction = None
                    confidence = None

                else:

                    # ======================================
                    # HEALTHY / DISEASE
                    # ======================================

                    if (
                        "healthy"
                        in prediction.lower()
                    ):

                        status = (
                            "Healthy Crop"
                        )

                    else:

                        status = (
                            "Disease Detected"
                        )

                    result = (
                        "AI analysis completed "
                        "successfully!"
                    )

            except Exception as e:

                print(
                    "Disease prediction error:",
                    e
                )

                status = "Error"

                result = (
                    "Prediction error: "
                    f"{str(e)}"
                )

        else:

            result = (
                "Please select a crop image."
            )

    return render_template(
        "disease.html",
        result=result,
        image_path=image_path,
        prediction=prediction,
        confidence=confidence,
        status=status
    )


# =========================================================
# YIELD PREDICTION
# =========================================================

@app.route(
    "/yield-prediction",
    methods=["GET", "POST"]
)
def yield_prediction():

    prediction = None
    total_production = None
    error = None

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

    if request.method == "POST":

        try:

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

            predicted_yield = float(
                yield_model.predict(
                    input_data
                )[0]
            )

            predicted_yield = max(
                0,
                predicted_yield
            )

            total_production = (
                predicted_yield *
                area
            )

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

    return render_template(
        "yield.html",
        prediction=prediction,
        total_production=total_production,
        error=error,
        form_data=form_data
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )