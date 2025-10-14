from flask import Flask, request, render_template, send_file, jsonify
from processing.image_processor import process_image
import threading
import io
import zipfile
import os
import tempfile
from threading import Lock

app = Flask(__name__)

# Global progress and lock for thread safety
progress = {
    "current": 0,
    "total": 1,
    "done": True,
    "error": None,
    "output_zip": None
}
progress_lock = Lock()

def update_progress(**kwargs):
    """Thread-safe way to update the global progress dictionary."""
    with progress_lock:
        progress.update(kwargs)

def process_zip(zip_file_stream):
    """Handles background image processing and ZIP creation."""
    update_progress(current=0, total=1, done=False, error=None, output_zip=None)
    print("🔧 Background thread started for processing ZIP...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded ZIP file to temporary location
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file_stream.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
                images = [
                    os.path.join(temp_dir, fname)
                    for fname in zip_ref.namelist()
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))
                ]

            if not images:
                raise RuntimeError("No valid image files found in uploaded ZIP.")

            update_progress(total=len(images))
            print(f"📸 Found {len(images)} images to process.")

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            for idx, img_path in enumerate(images):
                try:
                    with open(img_path, "rb") as f:
                        processed_img = process_image(f)

                        # ✅ process_image must return a BytesIO or readable stream
                        if hasattr(processed_img, "read"):
                            out_path = os.path.join(output_dir, os.path.basename(img_path))
                            with open(out_path, "wb") as out_f:
                                out_f.write(processed_img.read())
                        else:
                            raise TypeError("process_image() must return a BytesIO or readable file-like object.")
                    
                    print(f"✅ Processed {idx + 1}/{len(images)}: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")

                update_progress(current=idx + 1)

            # Double-check output directory
            output_files = os.listdir(output_dir)
            if not output_files:
                raise RuntimeError("No processed images were generated. Check process_image().")

            print(f"📦 Zipping {len(output_files)} processed images...")

            # Create in-memory ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as out_zip:
                for file_name in output_files:
                    file_path = os.path.join(output_dir, file_name)
                    out_zip.write(file_path, arcname=file_name)

            zip_buffer.seek(0)
            update_progress(done=True, output_zip=zip_buffer)
            print("🎉 ZIP ready and stored in memory.")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        update_progress(error=str(e), done=True)

@app.route('/')
def home():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/process_folder', methods=['POST'])
def process_folder():
    """Handle the uploaded ZIP and start a background thread."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    zip_file = request.files['file']
    if zip_file.filename == '':
        return "No file selected", 400

    # ✅ Fix: read bytes before starting the thread (Flask closes stream after response)
    file_bytes = zip_file.read()

    threading.Thread(target=process_zip, args=(io.BytesIO(file_bytes),)).start()
    return jsonify({"message": "Batch processing started."}), 202

@app.route('/progress')
def get_progress():
    """Return the current processing progress."""
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
    """Serve the processed ZIP file when available."""
    with progress_lock:
        if not progress.get("done"):
            return "Processing not finished yet.", 400
        if not progress.get("output_zip"):
            return "No processed output available.", 400
        zip_buffer = progress["output_zip"]

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="processed_images.zip"
    )

if __name__ == '__main__':
    app.run(debug=True)
