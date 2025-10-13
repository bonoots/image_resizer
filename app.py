from flask import Flask, request, render_template, send_file, jsonify
from processing.image_processor import process_image
import threading
import io
import zipfile
import os
import tempfile

app = Flask(__name__)

# Global variable to store progress
progress = {"current": 0, "total": 1, "done": True, "error": None}

def process_zip(zip_file_stream):
    progress["current"] = 0
    progress["done"] = False
    progress["error"] = None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded zip to a temp file
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file_stream.read())

            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                images = [os.path.join(temp_dir, fname) for fname in zip_ref.namelist() if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
            
            progress["total"] = len(images)
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            for idx, img_path in enumerate(images):
                try:
                    with open(img_path, "rb") as f:
                        processed_img = process_image(f)
                        out_path = os.path.join(output_dir, os.path.basename(img_path))
                        with open(out_path, "wb") as out_f:
                            out_f.write(processed_img.read())
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                progress["current"] = idx + 1

        progress["done"] = True
    except Exception as e:
        progress["error"] = str(e)
        progress["done"] = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_folder', methods=['POST'])
def process_folder():
    if 'file' not in request.files:
        return "No file uploaded", 400
    zip_file = request.files['file']
    if zip_file.filename == '':
        return "No file selected", 400

    # Start background processing thread
    threading.Thread(target=process_zip, args=(zip_file.stream,)).start()
    return jsonify({"message": "Batch processing started."}), 202

@app.route('/progress')
def get_progress():
    if progress["total"] == 0:
        percent = 0
    else:
        percent = int((progress["current"] / progress["total"]) * 100)
    return jsonify({
        "percent": percent,
        "current": progress["current"],
        "total": progress["total"],
        "done": progress["done"],
        "error": progress["error"]
    })

if __name__ == '__main__':
    app.run(debug=True)
