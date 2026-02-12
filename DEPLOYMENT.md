# Deployment Guide

This application is containerized with Docker, making it easy to deploy to any cloud provider that supports Docker containers (Render, Railway, AWS ECS, Google Cloud Run, Azure App Service, etc.).

## Method 1: Deploy via Docker Hub (Recommended)

Since this project contains large model files, building the Docker image locally and pushing it to Docker Hub is often more reliable than letting a cloud provider build it from Git (which might timeout ensuring Git LFS downloads).

### 1. Prerequisites
-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
-   A [Docker Hub](https://hub.docker.com/) account (free).

### 2. Build and Push the Image

1.  **Log in to Docker Hub** in your terminal:
    ```bash
    docker login
    ```

2.  **Build the image** (replace `youralgorithm` with your Docker Hub username):
    ```bash
    # IMPORTANT: Use linux/amd64 platform for cloud compatibility (most servers are x86, not ARM like M1 Macs)
    docker build --platform linux/amd64 -t yourusername/alzheimers-classifier:v1 .
    ```

3.  **Push the image**:
    ```bash
    docker push yourusername/alzheimers-classifier:v1
    ```

### 3. Deploy to Cloud (Example: Render.com)

1.  Create a **New Web Service** on [Render](https://render.com/).
2.  Select **"Deploy an existing image from a registry"**.
3.  Enter your image URL: `yourusername/alzheimers-classifier:v1`
4.  Render will pull the image and start it.
    -   **Port**: The app runs on port `5000`. Render should detect this, or you can set the `PORT` environment variable to `5000`.
    -   **RAM**: This app loads large models. Select a plan with at least **2GB-4GB RAM**.

---

## Method 2: Deploy to a Virtual Machine (AWS EC2 / DigitalOcean)

If you have a Linux server (Ubuntu/Debian) with Docker installed:

1.  **SSH into your server**.
2.  **Pull the image**:
    ```bash
    docker pull yourusername/alzheimers-classifier:v1
    ```
3.  **Run the container**:
    ```bash
    docker run -d -p 80:5000 --restart always --name alzheimers-app yourusername/alzheimers-classifier:v1
    ```
    (This runs the app in the background, mapping server port 80 to container port 5000).

---

## Troubleshooting

-   **Memory Errors**: If the app crashes on startup, the server likely ran out of RAM loading EfficientNet or the Generator models. Upgrade to a larger instance type.
-   **Platform Errors**: If you see "exec format error", you likely built the Docker image on an M1/M2 Mac without specifying `--platform linux/amd64`. Re-build using the command in Method 1.
