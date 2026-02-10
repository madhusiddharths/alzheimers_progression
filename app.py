import os
import uuid
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision.utils import save_image

import sys
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from pipeline import predict_full_progression, identify_stage, load_classifier_model
except ImportError as e:
    print(f"Error importing from pipeline: {e}")
    sys.exit(1)

UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = os.path.join('static', 'generated')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    force_stage = request.form.get('force_stage') # Get force_stage from form data

    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    original_filename = f"{unique_id}_{filename}"
    original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(original_filepath)
    print(f"File saved to: {original_filepath}")

    start_stage = None
    progression_results = {}

    try:
        print(f"Running prediction pipeline for: {original_filepath}")
        
        # If forced stage is provided, use it
        if force_stage and force_stage in ['Non Demented', 'Very mild Dementia', 'Mild Dementia']:
             print(f"Forcing start stage to: {force_stage}")
             progression_results = predict_full_progression(original_filepath, start_stage_override=force_stage)
             start_stage = force_stage # Display as forced
        else:
             progression_results = predict_full_progression(original_filepath)
             # Identify stage separately for display if not forced (or if predict returned empty)
             start_stage = identify_stage(original_filepath) or "Identification Failed"

        print(f"Pipeline returned {len(progression_results)} generated stages.")
        
    except Exception as e:
        print(f"Pipeline Error: {e}")
        return f"An error occurred during pipeline processing: {e}", 500

    generated_image_paths = {}
    if progression_results:
        for stage_name, image_tensor in progression_results.items():
            try:
                filename_stage = stage_name.replace(" ", "_").lower()
                generated_filename = f"{unique_id}_predicted_{filename_stage}.png"
                generated_filepath_abs = os.path.join(app.config['GENERATED_FOLDER'], generated_filename)
                save_image(image_tensor.cpu() * 0.5 + 0.5, generated_filepath_abs)
                generated_image_paths[stage_name] = os.path.join('generated', generated_filename)
                print(f"Saved generated image: {generated_filepath_abs}")
            except Exception as e:
                print(f"ERROR saving image for stage {stage_name}: {e}")

    static_original_filename = f"{unique_id}_original_{filename}"
    static_original_filepath_abs = os.path.join(app.config['GENERATED_FOLDER'], static_original_filename)
    original_image_path_for_template = None
    try:
        img = Image.open(original_filepath)
        img.save(static_original_filepath_abs)
        original_image_path_for_template = os.path.join('generated', static_original_filename)
    except Exception as e:
         print(f"Error copying original image to static: {e}")

    return render_template(
        'results.html',
        start_stage=start_stage,
        original_image_path=original_image_path_for_template,
        generated_images=generated_image_paths
    )

if __name__ == '__main__':
    load_classifier_model()
    app.run(debug=False) 