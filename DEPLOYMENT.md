# Deployment Guide

This application uses large AI model files that are **excluded from Git** to keep the repository lightweight. This makes deployment slightly different from standard web apps.

## 🐳 Docker Deployment (Recommended)

The most reliable way to deploy is to build a Docker image **locally** (where you have the model files) and then push that image to a container registry (like Docker Hub).

### Prerequisite: Large Files
Ensure you have the following directories populated locally:
-   `vgg19/` containing `vgg19_best_model.keras`
-   `gans/` containing your `.pth` generator checkpoints

### Step 1: Build the Image

Run this command in the project root. This copies your local code **and** the model files into the image.

```bash
docker build -t alzheimers-app .
# This might take a few minutes as it installs TensorFlow and PyTorch
```

### Step 2: Test Locally

Verify the container works before deploying:

```bash
docker run -p 5000:5000 alzheimers-app
```
Visit `http://localhost:5000`.

### Step 3: Push to Registry (e.g., Docker Hub)

1.  Log in to Docker Hub:
    ```bash
    docker login
    ```
2.  Tag the image:
    ```bash
    docker tag alzheimers-app <your-dockerhub-username>/alzheimers-app:latest
    ```
3.  Push the image:
    ```bash
    docker push <your-dockerhub-username>/alzheimers-app:latest
    ```

### Step 4: Deploy to Cloud (e.g., Render, Railway, AWS)

On your cloud provider:
1.  Select **"Deploy from Docker Hub"** or **"Container Registry"**.
2.  Enter the image name: `<your-dockerhub-username>/alzheimers-app:latest`.
3.  Set the **Internal Port** to `5000`.
4.  Deploy!

---

## ☁️ Why Git Deployment Fails

If you try to connect this GitHub repository directly to a service like Render or Heroku:
1.  The service clones the repo.
2.  It sees `.gitignore` excludes `vgg19/` and `gans/`.
3.  The build fails (or the app crashes at runtime) because **the model files are missing**.

To fix this for Git deployment, you would need to use **Git LFS (Large File Storage)** to track the model files, but this can use up your storage quota quickly. The Docker method above avoids this issues.
