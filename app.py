from flask import Flask, request, render_template
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ===== MODEL PATH =====
model = load_model("model/tomato_cnn_model.h5")

# ===== CLASS LABELS (FROM train_gen.class_indices) =====
idx_to_class = {
0:'Tomato Bacterial spot',
1:'Tomato Early blight',
2:'Tomato Late blight',
3:'Tomato Leaf Mold',
4:'Tomato Septoria leaf spot',
5:'Tomato Spider mites Two spotted spider mite',
6:'Tomato Target Spot',
7:'Tomato Tomato mosaic virus',
8:'Tomato Tomato Yellow Leaf Curl Virus',
9:'Tomato healthy'
}

# ===== CREATE STATIC FOLDER =====
UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ===== IMAGE PREDICTION FUNCTION =====
def predict_image(img_path):

    img = image.load_img(img_path, target_size=(224,224))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    disease = idx_to_class[pred_index]

    return disease, round(confidence*100,2)


# ===== HOME PAGE =====
@app.route("/", methods=["GET","POST"])
def index():

    prediction = ""
    confidence = ""
    img_path = ""

    if request.method == "POST":

        file = request.files["image"]

        filename = secure_filename(file.filename)

        img_path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(img_path)

        prediction, confidence = predict_image(img_path)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path
    )


# ===== RUN APP =====
if __name__ == "__main__":
    app.run(debug=True);