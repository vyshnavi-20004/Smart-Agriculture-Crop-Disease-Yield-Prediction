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
    BASE_DIR, "models", "crop_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "models", "class_names.json"
)

YIELD_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "yield_prediction_model.joblib"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# TensorFlow configuration
# =========================================================

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# =========================================================
# LOAD IMAGE VALIDATION MODEL
# =========================================================

print("Loading image validation model...")

validation_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True
)

print("Image validation model loaded successfully!")


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
# IMAGE VALIDATION FUNCTION
# =========================================================

def validate_crop_image(filepath):

    try:

        # Open image
        img = Image.open(filepath).convert("RGB")

        # Resize for MobileNetV2
        img = img.resize((224, 224))

        # Convert to NumPy
        img_array = np.array(img, dtype=np.float32)

        # MobileNetV2 preprocessing
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # ImageNet prediction
        predictions = validation_model.predict(
            img_array,
            verbose=0
        )

        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
            predictions,
            top=5
        )[0]

        print("Image validation predictions:")

        for item in decoded:
            print(
                item[1],
                round(float(item[2]) * 100, 2),
                "%"
            )

        # -------------------------------------------------
        # Classes that clearly indicate a non-crop image
        # -------------------------------------------------

        invalid_keywords = [

            # People
            "person",
            "man",
            "woman",
            "boy",
            "girl",
            "groom",
            "bride",

            # Animals
            "dog",
            "cat",
            "bird",
            "horse",
            "cow",
            "sheep",
            "goat",
            "rabbit",
            "monkey",
            "elephant",
            "tiger",
            "lion",
            "bear",

            # Vehicles
            "car",
            "taxi",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "airliner",
            "airplane",
            "ship",
            "boat",

            # Electronics
            "cellphone",
            "mobile",
            "laptop",
            "computer",
            "television",
            "monitor",
            "keyboard",
            "mouse",

            # Buildings / places
            "building",
            "church",
            "mosque",
            "palace",
            "castle",
            "restaurant",
            "shop",

            # Furniture / objects
            "chair",
            "table",
            "sofa",
            "bed",
            "book",
            "pencil",
            "pen",
            "bottle",
            "cup",
            "backpack",
            "umbrella",
            "shoe",
            "watch"
        ]

        # Check top predictions
        for _, label, confidence in decoded:

            label_lower = label.lower()

            if confidence >= 0.20:

                for keyword in invalid_keywords:

                    if keyword in label_lower:

                        return False

        # -------------------------------------------------
        # Check whether ImageNet detected a plant
        # -------------------------------------------------

        plant_keywords = [

            "plant",
            "leaf",
            "tree",
            "flower",
            "daisy",
            "sunflower",
            "rose",
            "corn",
            "pot",
            "cucumber",
            "mushroom",
            "acorn",
            "pine",
            "fir",
            "fig",
            "strawberry",
            "orange",
            "lemon",
            "apple",
            "banana",
            "pineapple",
            "pomegranate",
            "grape",
            "jackfruit",
            "coffee"
        ]

        plant_found = False

        for _, label, confidence in decoded:

            label_lower = label.lower()

            for keyword in plant_keywords:

                if keyword in label_lower and confidence >= 0.05:

                    plant_found = True
                    break

            if plant_found:
                break

        # -------------------------------------------------
        # Additional green-pixel check
        # -------------------------------------------------

        original = Image.open(filepath).convert("RGB")
        original = original.resize((256, 256))

        image_array = np.array(original)

        red = image_array[:, :, 0].astype(np.int16)
        green = image_array[:, :, 1].astype(np.int16)
        blue = image_array[:, :, 2].astype(np.int16)

        green_pixels = (
            (green > red * 1.05) &
            (green > blue * 1.02) &
            (green > 50)
        )

        green_ratio = np.mean(green_pixels)

        print(
            "Green pixel ratio:",
            round(float(green_ratio) * 100, 2),
            "%"
        )

        # -------------------------------------------------
        # Final validation decision
        # -------------------------------------------------

        if plant_found:

            return True

        # Allow images with a reasonable amount of green
        # because some leaves may not be classified correctly
        if green_ratio >= 0.08:

            return True

        return False

    except Exception as e:

        print("Image validation error:", e)

        return False


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


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

@app.route("/disease", methods=["GET", "POST"])
def disease():

    result = None
    image_path = None
    prediction = None
    confidence = None
    status = None

    if request.method == "POST":

        image = request.files.get("crop-image")

        if image and image.filename != "":

            filename = image.filename

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(filepath)

            image_path = "/uploads/" + os.path.basename(filepath)

            try:

                # ==========================================
                # STEP 1: VALIDATE IMAGE
                # ==========================================

                is_valid_crop = validate_crop_image(filepath)

                if not is_valid_crop:

                    status = "Invalid Image"

                    result = (
                        "This image does not appear to be a "
                        "crop or leaf image. Please upload a "
                        "valid crop leaf image."
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

                img = Image.open(filepath).convert("RGB")

                img = img.resize((160, 160))

                img_array = np.array(
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
                    np.argmax(predictions[0])
                )

                confidence = float(
                    predictions[0][predicted_index] * 100
                )

                prediction = class_names[
                    predicted_index
                ]

                # ==========================================
                # CONFIDENCE CHECK
                # ==========================================

                if confidence < 60:

                    status = "Uncertain Image"

                    result = (
                        "The image could not be classified "
                        "with sufficient confidence. "
                        "Please upload a clear crop leaf image."
                    )

                    prediction = None
                    confidence = None

                else:

                    # ======================================
                    # HEALTHY / DISEASE
                    # ======================================

                    if "healthy" in prediction.lower():

                        status = "Healthy Crop"

                    else:

                        status = "Disease Detected"

                    result = (
                        "AI analysis completed successfully!"
                    )

            except Exception as e:

                status = "Error"

                result = (
                    f"Prediction error: {str(e)}"
                )

        else:

            result = "Please select a crop image."

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
                predicted_yield * area
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