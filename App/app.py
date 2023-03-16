from flask import Flask, jsonify, request
from process_image import get_emotion

classes = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'surprised': 3,
    'fearful': 4,
    'disgusted': 5,
    'angry': 6,
    'contempt': 7
}

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