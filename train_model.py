import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset/raw/color"
MODEL_PATH = "models/crop_disease_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5

# ==============================
# CHECK DATASET
# ==============================

print("\nChecking dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

print("Dataset found!")
print("Loading images...")

# ==============================
# LOAD DATASET
# ==============================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nNumber of classes:", num_classes)
print("Classes:")
for i, name in enumerate(class_names):
    print(i, "->", name)

# ==============================
# SAVE CLASS NAMES
# ==============================

os.makedirs("models", exist_ok=True)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

print("\nClass names saved to:", CLASS_NAMES_PATH)

# ==============================
# PERFORMANCE OPTIMIZATION
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==============================
# MOBILE NET V2
# ==============================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers
base_model.trainable = False

# ==============================
# BUILD MODEL
# ==============================

inputs = layers.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully!")

model.summary()

# ==============================
# TRAIN
# ==============================

print("\n==============================")
print("STARTING TRAINING")
print("==============================\n")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS
)

# ==============================
# SAVE MODEL
# ==============================

model.save(MODEL_PATH)

print("\n==============================")
print("TRAINING COMPLETED!")
print("==============================")
print("Model saved to:", MODEL_PATH)
print("Classes saved to:", CLASS_NAMES_PATH)

# ==============================
# FINAL ACCURACY
# ==============================

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print("\nFinal Training Accuracy:",
      round(final_train_accuracy * 100, 2), "%")

print("Final Validation Accuracy:",
      round(final_val_accuracy * 100, 2), "%")