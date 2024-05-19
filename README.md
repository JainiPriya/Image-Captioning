# Image Captioning Service on Google Kubernetes Engine (GKE)

## Project Overview
This project develops an end-to-end image captioning service that uses deep learning models to automatically generate textual descriptions for images. The service combines computer vision and natural language processing techniques, employing a CNN-LSTM architecture for generating captions. It is deployed on Google Kubernetes Engine (GKE) to leverage cloud scalability and robustness.

## Features
- **Deep Learning Model**: Uses Xception for feature extraction and LSTM for generating captions.
- **Gradio Interface**: Provides a user-friendly web interface for real-time image captioning.
- **GKE Deployment**: Ensures scalable and reliable service availability using Kubernetes.

## Prerequisites
- Google Cloud account
- Docker installed on your local machine
- Kubernetes CLI (kubectl)
- Python 3.8 or higher
- Access to Google Kubernetes Engine

## Setup Instructions

### 1. Unzip the Repository
```bash
unzip Code.zip
cd Code
```
### 2. Run the image-captioning-training.ipynb file on colab or local to create the training model and save the final model to the inference folder

### 3. Navigate to GCP cloud shell and upload all the files(inference folder and pvc.yaml) necessary

### 4. Create a Kubernetes cluster as explained in the report and execute all the instructions to deploy the application
