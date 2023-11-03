from flask import Flask, render_template, request,jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model_path = "E:\pymodel\model\modelMLtest.h5"
model = load_model(model_path)

@app.route('/', methods=['GET'])
def index():
    return render_template('testml.html')

@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # โหลดรูปภาพและปรับขนาด
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # ทำนายผลลัพธ์ด้วยโมเดล
    yhat = model.predict(image)
    class_index = np.argmax(yhat)
    
    # รายชื่อคลาส
    label_names = ['Astrophytum', 'Mammillaria', 'Melocactus', 'cactus']
    classification = label_names[class_index]

    return render_template('testml.html', prediction=classification)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)