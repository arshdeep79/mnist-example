from flask import Flask, request, jsonify
import yaml 
import libs.modelInterface as modelInterface
import threading
import time
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/run/train", methods=["POST"])
def run_train():
    if 'config' not in request.files:
        return jsonify({'error': 'Bad Request - Missing "config" file'}), 400 
    
    file = request.files['config']
    parsedFile = yaml.safe_load(file)
    
    threading.Thread(target=modelInterface.train, args=([parsedFile])).start()
    return jsonify({"status":"success", "message":"Training started please check neptune dashboard for progress"}) 


@app.route("/run/inference", methods=["POST"])
def run_inference():
    if 'image' not in request.files:
        return jsonify({'error': 'Bad Request - Missing "image" file'}), 400 
    
    file = request.files['image']
    image = Image.open(BytesIO(file.read()))
    prediction = modelInterface.infer(image) 

    return jsonify({"status":"success", "prediction":prediction}) 


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)