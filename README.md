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

## Technology Stack

-   **Backend**: Flask (Python)
-   **AI/ML**:
    -   **Classification**: VGG19 (Transfer Learning) with TensorFlow/Keras.
    -   **Generation**: Deep Convolutional GANs (DCGAN) with PyTorch.
-   **Frontend**: HTML5, CSS3, JavaScript.
-   **Containerization**: Docker.

## Project Structure

```
├── app.py                 # Main Flask application
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── src/
│   ├── pipeline.py        # Core logic for prediction and generation
│   └── ...
├── gans/                  # [Large File] PyTorch GAN Generator models
├── vgg19/                 # [Large File] VGG19 Classification model
├── templates/             # HTML templates
└── static/                # CSS, JS, and generated images
```

## Setup & Installation

### Prerequisites

-   Python 3.9+
-   Docker (optional, recommended for deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/madhusiddharths/alzheimers_progression.git
cd alzheimers_progression
```

### 2. Download Model Files
> [!IMPORTANT]
> Because the model files are large, they are **NOT** stored in this GitHub repository. You must have the `vgg19/` and `gans/` directories populated locally.

Ensure your directory structure looks like this:
-   `vgg19/vgg19_best_model.keras`
-   `gans/madhu/generator.pth` (and other stage checkpoints)

### 3. Run Locally (Python)

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

### 4. Run with Docker

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on building and running the Docker container.

## Data Sources

-   **OASIS MRI Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
-   **Augmented Alzheimer MRI Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

## License
[MIT License](LICENSE)
