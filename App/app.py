from flask import Flask, jsonify, request
from process_image import get_emotion

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'message': 'Backend is working!'})


@app.route('/image', methods=['POST'])
def image():
    image = request.files['image']
    emotion = get_emotion(image)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run()