from flask import Flask, request, render_template,jsonify
import numpy as np
import os
import tensorflow as tf
import shap
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import base64



app = Flask(__name__)

# Load the model
model = load_model('forestFireDetect_Final.h5', compile=False)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0][0]
    percentage = preds *100 if preds>0.5 else(1-preds)*100
    
    if preds >0.5:
        return f"Fire-Not-detected✅ ({percentage:.2f}%) "
    else:
        return f"Fire-detected🔥 ({percentage:.2f}%)" 

@app.route('/predict',methods=['GET', 'POST'])
def predict_fire():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f.save(file_path)

        preds = model_predict(file_path, model)
        shap_img = create_shap_explanation(file_path, model)
        print(f"Prediction: {preds}, SHAP Image: {shap_img}")
        return jsonify({
        'prediction': preds,
        'shap_image': shap_img })

explainer_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
def preprocess_img(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy().astype(np.float32)

def create_shap_explanation(img_path, model):
    try:
        print("✅ create_shap_explanation running")

        # Load and preprocess the image
        x = preprocess_img(img_path)


        def f(x_input):
            tmp = x_input.copy()
            return model(tmp)
        
        masker_blur = shap.maskers.Image("blur(256,256)", shape=(256, 256, 3))

        explainer = shap.Explainer(f, masker_blur)
        shap_values = explainer(x[np.newaxis, :, :, :], max_evals=1000)


        plt.title('SHAP Explainer')
        shap.image_plot(shap_values,show=False)

        # Save SHAP plot
        temp_path = os.path.join(os.getcwd(), "temp_shap_plot.png")
        plt.savefig(temp_path,bbox_inches='tight', pad_inches=0.1)
        plt.close()

        with open(temp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        os.remove(temp_path)

        print("✅ SHAP explanation complete")
        return encoded

    except Exception as e:
        import traceback
        print("🔥 SHAP ERROR:")
        traceback.print_exc()
        return None
    finally:
        tf.keras.backend.clear_session()



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/alert', methods=['POST'])
def alert():
    address = request.form['address']
    return jsonify({"status": "success", "message": f"Alert sent for {address}"})


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
