# Alzheimer's Disease Progression Analysis & Prediction

A deep learning application that analyzes MRI scans to detect Alzheimer's disease stages and generates visual progressions of the disease using Generative Adversarial Networks (GANs).

![Project Banner](static/img/banner_placeholder.png) <!-- Ideally add a screenshot here -->

## Features

-   **Disease Stage Detection**: Classifies MRI scans into 4 stages:
    -   Non Demented
    -   Very Mild Dementia
    -   Mild Dementia
    -   Moderate Dementia
-   **Progression Visualization**: Generates synthetic MRI images showing how the brain might look as the disease advances to subsequent stages.
-   **User-Friendly Interface**: a web interface for uploading scans and viewing results.
-   **Mac Optimized**: Native support for Mac GPU (Metal Performance Shaders) for accelerated training and inference.

## Technology Stack

-   **Backend**: Flask (Python)
-   **AI/ML**:
    -   **Classification**: **EfficientNetB4** (Transfer Learning) with **PyTorch**.
    -   **Generation**: Deep Convolutional GANs (DCGAN) with **PyTorch**.
    -   **Hardware Acceleration**: Apple Metal (MPS) support.
-   **Frontend**: HTML5, CSS3, JavaScript.
-   **Containerization**: Docker.

## Project Structure

```
├── app.py                     # Main Flask application
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── src/
│   ├── pipeline.py            # Core logic for prediction and generation
│   └── ...
├── model_classifier/          # Classifier training script and model
│   ├── train_classifier.py    # PyTorch training script
│   └── efficientnet_b4_pytorch.pth  # Trained classifier model [Git LFS]
├── gans/                      # [Large Files] PyTorch GAN Generator models
├── templates/                 # HTML templates
└── static/                    # CSS, JS, and generated images
```

## Setup & Installation

### Prerequisites

-   Python 3.9+
-   Docker (optional, recommended for deployment)
-   [Git LFS](https://git-lfs.github.com/) (required to download model weights)

### 1. Clone the Repository

```bash
git clone https://github.com/madhusiddharths/alzheimers_progression.git
cd alzheimers_progression
git lfs pull  # Download large model files
```

### 2. Run Locally (Python)

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    python app.py
    ```
4.  Open `http://127.0.0.1:5000` in your browser.

### 3. Run with Docker

Build the container:
```bash
docker build -t alzheimers-classifier .
```

Run the container:
```bash
docker run -p 5000:5000 alzheimers-classifier
```
*Note: If port 5000 is in use (common on macOS AirPlay), run on port 5001:*
```bash
docker run -p 5001:5000 alzheimers-classifier
```

## Data Sources

-   **OASIS MRI Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
-   **Augmented Alzheimer MRI Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

## License
[MIT License](LICENSE)
