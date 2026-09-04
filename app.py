import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("cnn_model.keras")

# Class names
class_names = ["NO", "YES"]


def predict_image(image):
    if image is None:
        return "Please upload an image."

    # Resize image
    image = image.resize((128, 128))

    # Convert to numpy array
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image, verbose=0)[0]

    # Get predicted class
    predicted_class = np.argmax(prediction)

    if predicted_class == 1:
        return "YES — Brain Tumor Detected"
    else:
        return "NO — No Brain Tumor Detected"


# -----------------------------
# Create the user interface
# -----------------------------

with gr.Blocks(
    theme=gr.themes.Soft()
) as app:

    gr.Markdown(
        """
        # 🧠 Brain Tumor Image Classifier

        ### Upload a brain scan image and let the CNN classify it.

        This application uses a Convolutional Neural Network (CNN)
        trained to classify images into **YES** or **NO**.
        """
    )

    gr.Markdown("---")

    with gr.Row():

        # LEFT SIDE — IMAGE UPLOAD
        with gr.Column():

            gr.Markdown("### 📤 Upload Image")

            image_input = gr.Image(
                type="pil",
                label="Brain Scan"
            )

            predict_button = gr.Button(
                "🔍 Predict",
                variant="primary"
            )

        # RIGHT SIDE — RESULT
        with gr.Column():

            gr.Markdown("### 🧾 Prediction")

            result = gr.Textbox(
                label="Result",
                placeholder="Your prediction will appear here...",
                lines=2
            )

    gr.Markdown("---")

    gr.Markdown(
        """
        ### ℹ️ How to use

        1. Upload an image.
        2. Click **Predict**.
        3. The CNN will classify the image.

        **Note:** This is a machine-learning project for educational
        purposes and is not a medical diagnostic tool.
        """
    )

    # Connect button to prediction function
    predict_button.click(
        fn=predict_image,
        inputs=image_input,
        outputs=result
    )


# Launch
app.launch(share=True)