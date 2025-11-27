# 🧠 Brain Tumor Detection System

A deep learning-based web application for detecting and classifying brain tumors from MRI images using Convolutional Neural Networks (CNN) with transfer learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [API Endpoints](#api-endpoints)
- [Supported Image Formats](#supported-image-formats)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an intelligent brain tumor detection system that can classify MRI images into four categories:
- **Pituitary Tumor**
- **Glioma Tumor**
- **Meningioma Tumor**
- **No Tumor**

The system uses VGG16 (a pre-trained CNN architecture) with transfer learning to achieve high accuracy in tumor detection and classification. The application features a user-friendly web interface built with Flask, allowing users to upload MRI images and receive instant predictions with confidence scores.

## ✨ Features

- 🔍 **Automatic Tumor Detection**: Upload MRI images and get instant tumor detection results
- 📊 **Multi-Class Classification**: Classifies tumors into 4 categories (pituitary, glioma, meningioma, no tumor)
- 📈 **Confidence Scores**: Displays prediction confidence for each classification
- 🖼️ **Image Visualization**: View uploaded images alongside detection results
- 🎨 **Modern Web Interface**: Clean and responsive UI built with Bootstrap
- ⚡ **Real-time Processing**: Fast inference using optimized TensorFlow models
- 🔒 **Secure File Handling**: Validates file types and handles uploads securely

## 🛠️ Technology Stack

### Backend
- **Python 3.x**
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning framework
- **Pillow (PIL)** - Image processing
- **NumPy** - Numerical operations

### Frontend
- **HTML5/CSS3**
- **Bootstrap 5** - Responsive UI framework
- **Jinja2** - Template engine

### Deep Learning
- **VGG16** - Pre-trained CNN model for transfer learning
- **Transfer Learning** - Fine-tuning pre-trained models

## 🏗️ Model Architecture

The model is built using **VGG16** as the base architecture with transfer learning:

1. **Base Model**: VGG16 pre-trained on ImageNet
   - Input shape: (128, 128, 3)
   - Last 3 layers are fine-tuned (trainable)
   - Remaining layers are frozen

2. **Custom Layers**:
   - Flatten layer
   - Dropout (0.3) for regularization
   - Dense layer (128 neurons, ReLU activation)
   - Dropout (0.2)
   - Output layer (4 neurons, softmax activation)

3. **Training Configuration**:
   - Optimizer: Adam (learning rate: 0.0001)
   - Loss function: Sparse categorical crossentropy
   - Batch size: 20
   - Epochs: 5

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Brain-Tumor-Detection
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
cd abc
pip install -r requirements.txt
```

### Step 4: Verify Model File

Ensure the trained model file exists at:
```
abc/models/model.h5
```

If the model file is missing, you'll need to train the model using the Jupyter notebook:
```
abc/brain_tumour_detection_using_deep_learning.ipynb
```

## 🚀 Usage

### Running the Application

1. **Navigate to the project directory:**
   ```bash
   cd abc
   ```

2. **Start the Flask server:**
   ```bash
   python main.py
   ```

3. **Open your web browser and navigate to:**
   ```
   http://localhost:5000
   ```

### Using the Web Interface

1. **Upload an MRI Image**:
   - Click "Select MRI Image" button
   - Choose an image file from your computer
   - Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF

2. **Get Prediction**:
   - Click "Upload and Detect" button
   - Wait for the model to process the image
   - View the results:
     - Tumor classification (or "No Tumor")
     - Confidence score percentage
     - Uploaded image preview

### Example Workflow

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Navigate to app directory
cd abc

# Run the application
python main.py

# Access the application at http://localhost:5000
```

## 📁 Project Structure

```
Brain-Tumor-Detection/
│
├── abc/                          # Main application directory
│   ├── main.py                   # Flask application entry point
│   ├── models/                   # Trained model directory
│   │   └── model.h5              # Pre-trained VGG16 model
│   ├── templates/                # HTML templates
│   │   └── index.html            # Main web interface
│   ├── uploads/                  # Uploaded images directory (auto-created)
│   ├── requirements.txt          # Python dependencies
│   ├── brain_tumour_detection_using_deep_learning.ipynb  # Model training notebook
│   └── README.md                 # Additional documentation
│
├── templates/                    # Alternative templates directory
│   └── index.html
│
└── README.md                     # This file
```

## 📊 Model Performance

The model achieved the following performance metrics on the test dataset:

- **Overall Accuracy**: ~95%
- **Precision**: 0.95 (macro average)
- **Recall**: 0.94 (macro average)
- **F1-Score**: 0.95 (macro average)

### Class-wise Performance:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Pituitary | 0.97 | 0.98 | 0.98 | 300 |
| Glioma | 0.93 | 0.90 | 0.91 | 300 |
| No Tumor | 0.95 | 1.00 | 0.97 | 405 |
| Meningioma | 0.93 | 0.91 | 0.92 | 306 |

## 🔌 API Endpoints

### GET `/`
- **Description**: Render the main upload page
- **Response**: HTML page with upload form

### POST `/`
- **Description**: Handle image upload and prediction
- **Request**: Multipart form data with image file
- **Response**: HTML page with prediction results

### GET `/uploads/<filename>`
- **Description**: Serve uploaded images
- **Response**: Image file

## 🖼️ Supported Image Formats

The application supports the following image formats:
- PNG (`.png`)
- JPEG/JPG (`.jpg`, `.jpeg`)
- BMP (`.bmp`)
- TIFF (`.tif`, `.tiff`)

**Note**: Images are automatically resized to 128x128 pixels during preprocessing to match the model's input requirements.

## ⚠️ Important Notes

1. **Medical Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

2. **Model Requirements**: The model file (`model.h5`) must be present in the `abc/models/` directory for the application to function.

3. **Production Deployment**: For production use:
   - Replace the default Flask secret key in `main.py`
   - Use a production WSGI server (e.g., Gunicorn, uWSGI)
   - Implement proper error handling and logging
   - Add authentication and rate limiting
   - Use HTTPS for secure connections

4. **Resource Requirements**: 
   - RAM: Minimum 4GB recommended
   - Disk: ~2GB for dependencies and model
   - GPU: Optional but recommended for faster inference

## 🐛 Troubleshooting

### Model Not Loading
- Verify that `model.h5` exists in `abc/models/` directory
- Check TensorFlow/Keras version compatibility
- Ensure the model file is not corrupted

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version (3.8+)
- Check virtual environment activation

### Upload Errors
- Verify file format is supported
- Check file size limits
- Ensure `uploads/` directory has write permissions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE) (or specify your license).

## 🙏 Acknowledgments

- VGG16 model architecture by the Visual Geometry Group, Oxford
- TensorFlow/Keras team for the deep learning framework
- Flask community for the web framework
- All contributors and researchers in medical imaging AI

## 📧 Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository.

---

**⚠️ Medical Disclaimer**: This software is provided for educational and research purposes only. It is not intended to be used for medical diagnosis, treatment, or as a substitute for professional medical advice. Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.
