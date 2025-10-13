import os
import tempfile
import zipfile
from pathlib import Path
from flask import Flask, request, render_template, send_file
from processing.image_processor import process_image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_folder', methods=['POST'])
def process_folder():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if not file.filename.lower().endswith('.zip'):
        return "Please upload a ZIP file containing your images.", 400

    # Use a temporary workspace for safe processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save and extract the uploaded ZIP
        zip_path = tmpdir / "uploaded.zip"
        file.save(zip_path)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(input_dir)
        except zipfile.BadZipFile:
            return "Invalid ZIP file.", 400

        # Process images recursively
        for img_path in input_dir.rglob('*'):
            if img_path.is_file() and allowed_file(img_path.name):
                rel = img_path.relative_to(input_dir)
                dest = (output_dir / rel).with_suffix('.jpg')
                dest.parent.mkdir(parents=True, exist_ok=True)
                
                # Pass the file path as a string, not a file object
                try:
                    processed_stream = process_image(str(img_path), str(dest))
                    if processed_stream is None:
                        print(f"skipping invalid file: {img_path.name}")
                        continue
                    with open(dest, 'wb') as out:
                        out.write(processed_stream.read())
                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")
                    continue

        # Package results into a ZIP
        output_zip_path = tmpdir / "enhanced_images.zip"
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(output_dir))

        return send_file(output_zip_path, as_attachment=True, download_name="enhanced_images.zip")

if __name__ == '__main__':
    # Bind to 0.0.0.0 for broader compatibility in containers/VMs; change as needed.
    app.run(host='0.0.0.0', port=3000, debug=True)
