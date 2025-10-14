from flask import Flask, request, render_template, send_file, jsonify
from processing.image_processor import process_image
import threading
import io
import zipfile
import os
import tempfile
from threading import Lock

app = Flask(__name__)

# Global progress dictionary and a lock for thread safety
progress = {"current": 0, "total": 1, "done": True, "error": None, "output_zip": None}
progress_lock = Lock()

def update_progress(**kwargs):
    """Thread-safe progress update."""
    with progress_lock:
        progress.update(kwargs)

def process_zip(zip_file_stream):
    """Handles image batch processing in background."""
    update_progress(current=0, total=1, done=False, error=None, output_zip=None)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded ZIP to a temp file
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file_stream.read())

            # Extract images
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                images = [
                    os.path.join(temp_dir, fname)
                    for fname in zip_ref.namelist()
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))
                ]

            update_progress(total=len(images))

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
                update_progress(current=idx + 1)

            # Create output ZIP
            output_zip_path = os.path.join(temp_dir, "processed_images.zip")
            with zipfile.ZipFile(output_zip_path, "w") as out_zip:
                for file_name in os.listdir(output_dir):
                    out_zip.write(os.path.join(output_dir, file_name), arcname=file_name)

            # Store in memory for download
            with open(output_zip_path, "rb") as f:
                zip_bytes = io.BytesIO(f.read())

            update_progress(done=True, output_zip=zip_bytes)
    except Exception as e:
        update_progress(error=str(e), done=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_folder', methods=['POST'])
def process_folder():
    """Handle ZIP upload and start background thread."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    zip_file = request.files['file']
    if zip_file.filename == '':
        return "No file selected", 400

    # ✅ FIX: Read file before background thread starts
    file_bytes = zip_file.read()

    threading.Thread(target=process_zip, args=(io.BytesIO(file_bytes),)).start()
    return jsonify({"message": "Batch processing started."}), 202

@app.route('/progress')
def get_progress():
    """Returns progress data for frontend polling."""
    with progress_lock:
        total = progress["total"]
        current = progress["current"]
        done = progress["done"]
        error = progress["error"]

    percent = 0 if total == 0 else int((current / total) * 100)
    return jsonify({
        "percent": percent,
        "current": current,
        "total": total,
        "done": done,
        "error": error
    })

@app.route('/download')
def download_result():
    """Serves processed ZIP if available."""
    with progress_lock:
        if not progress.get("done") or not progress.get("output_zip"):
            return "Processing not finished or no output available.", 400
        zip_bytes = progress["output_zip"]

    zip_bytes.seek(0)
    return send_file(
        zip_bytes,
        mimetype="application/zip",
        as_attachment=True,
        download_name="processed_images.zip"
    )

if __name__ == '__main__':
    app.run(debug=True)
