# 👕 VestiAI — AI Virtual Try-On

> **See any garment on you, live.**

VestiAI is an AI-powered virtual try-on platform that allows users to visualize garments such as T-shirts, shirts, jackets, kurtas, and dresses on themselves using computer vision, pose tracking, image processing, and deep learning.

The system provides a real-time virtual try-on experience by detecting the user's body pose, processing the selected garment, and intelligently fitting the garment onto the user's body.

---

## ✨ Features

- 🎥 **Real-Time Virtual Try-On**
- 👤 **Human Pose Detection & Tracking**
- 🧠 **AI-Based Garment Processing**
- 👕 **Garment Category Classification**
- 🖼️ **Garment Background Removal**
- 🎭 **Garment Mask Generation**
- 📐 **Perspective & Affine Transformation**
- 🔥 **Computer Vision Pipeline**
- 🤖 **Deep Learning-Based Try-On**
- 📸 **Camera-Based Live Preview**
- 📤 **Garment Image Upload**
- 🧪 **Sample Garment Generation**
- 📊 **System Readiness Monitoring**
- ⚡ **FastAPI Backend**
- 🎨 **Modern Responsive Web Interface**

---

# 🧠 How VestiAI Works

VestiAI follows a dual-pipeline architecture combining Computer Vision and Deep Learning.

```text
                    ┌──────────────────────┐
                    │       User           │
                    │  Camera / Garment    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Garment Upload     │
                    │   / Camera Input     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Background Removal   │
                    │ & Image Processing   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Garment Masking    │
                    │ & Classification     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Pose Detection     │
                    │   & Body Landmarks   │
                    └──────────┬───────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
              ┌──────────────┐    ┌─────────────────┐
              │ Real-Time CV │    │ Deep Learning   │
              │ Try-On       │    │ Try-On Pipeline │
              └──────┬───────┘    └────────┬────────┘
                     │                     │
                     └──────────┬──────────┘
                                ▼
                    ┌──────────────────────┐
                    │ Final Try-On Result  │
                    │     👕 + 👤          │
                    └──────────────────────┘

Deep Learning Pipeline

The deep-learning pipeline is designed for higher-quality and more realistic results.

Person Image + Garment Image
              ↓
         Preprocessing
              ↓
       Image Segmentation
              ↓
        Pose Information
              ↓
      Deep Learning Model
              ↓
       Image Generation
              ↓
   Photorealistic Try-On

🏗️ System Architecture

                         VestiAI
                            │
              ┌─────────────┴─────────────┐
              │                           │
           Frontend                    Backend
              │                           │
        Web Interface                  FastAPI
              │                           │
        Camera Input              Processing Services
              │                           │
        Garment Upload            AI / CV Pipeline
              │                           │
              └─────────────┬─────────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
           Computer Vision       Deep Learning
                  │                   │
           Pose Tracking       Generative Models
                  │                   │
           Image Warping       Image Synthesis
                  │                   │
                  └─────────┬─────────┘
                            │
                            ▼
                   Virtual Try-On Result

📁 Project Structure

VestiAI/
│
├── backend/
│   ├── API
│   ├── services
│   ├── processing
│   └── model adapters
│
├── frontend/
│   ├── components
│   ├── pages
│   ├── styles
│   └── assets
│
├── captures/
│   └── Camera captures
│
├── checkpoints/
│   └── Model checkpoints
│
├── configs/
│   └── Configuration files
│
├── datasets/
│   └── Dataset resources
│
├── garments/
│   └── Garment images
│
├── logs/
│   └── Application logs
│
├── models_cache/
│   └── Cached AI models
│
├── results/
│   └── Generated results
│
├── scripts/
│   └── Utility scripts
│
├── tests/
│   └── Test cases
│
├── uploads/
│   └── Uploaded garments
│
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── requirements-ml.txt
├── run.py
└── README.md

🛠️ Technology Stack

Frontend

* HTML5
* CSS3
* JavaScript
* Responsive Web UI
* Browser Camera API

Backend

* Python
* FastAPI
* REST APIs
* WebSocket communication

Computer Vision

* OpenCV
* Pose Estimation
* Image Segmentation
* Image Masking
* Affine Transformation
* Perspective Transformation
* Image Warping

Artificial Intelligence

* Deep Learning
* Computer Vision
* Generative AI
* Image Synthesis
* Diffusion-based models
* Pose-guided generation

Storage

* SQLite
* PostgreSQL support
* Local file storage

Development

* Python
* Git
* GitHub
* VS Code
* Virtual Environment

## 👕 How to Use

Launch VestiAI and open the web interface in a modern browser. Click **Start Live Try-On** and allow camera access when prompted. Upload a garment image such as a T-shirt, shirt, jacket, kurta, or dress using the **Upload a Garment** option. VestiAI automatically processes the garment by removing the background, generating a garment mask, and identifying the garment category. The system then detects the user's body pose and landmarks through computer vision and aligns the selected garment with the user's body using image transformation and warping techniques. Finally, the processed garment is rendered on the user to provide a real-time virtual try-on experience, while the deep-learning pipeline can be used for more realistic AI-generated try-on results.