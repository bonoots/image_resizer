from flask import Flask, request, render_template, send_file
from processing.image_processor import process_image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "Empty filename", 400

    processed_image = process_image(file.stream)
    return send_file(processed_image, mimetype='image/jpeg', as_attachment=True, download_name='enhanced.jpg')

if __name__ == '__main__':
    app.run(debug=True)
