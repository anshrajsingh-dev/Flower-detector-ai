# 🌸 Flower Detector AI

A browser-based AI application that identifies flower species from an uploaded image using a deep learning model exported to **ONNX**. The application performs inference entirely in the browser with **ONNX Runtime Web**, so no backend server is required.

---

## 📖 Overview

Flower Detector AI is an image classification project built to demonstrate how a deep learning model can be deployed directly in a web application.

Users can upload a flower image, and the application predicts the flower species while displaying useful information such as its meaning, interesting facts, and basic care instructions.

The project combines machine learning with modern web technologies to create a lightweight and interactive educational application.

---

## ✨ Features

* 🌼 Flower image classification
* 🧠 Deep learning model exported to ONNX
* ⚡ Browser-side inference using ONNX Runtime Web
* 📷 Image upload and preview
* 🌺 Displays flower information after prediction
* 🎨 Responsive and user-friendly interface
* 🔒 Images remain on the user's device (no server upload)

---

## 🌸 Supported Flowers (12-Universe Dataset)

The model is trained to recognize **12 distinct flower species** compiled from the Oxford dataset:

* Bluebell | Buttercup | Coltsfoot
* Cowslip | Crocus | Daffodil
* Daisy | Fritillary | Iris
* Lily | Pansy | Sunflower

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Jupyter Notebook
* ONNX

### Frontend

* HTML5
* CSS3
* JavaScript (ES6)

### AI Runtime

* ONNX Runtime Web

---

## 📁 Project Structure

```text
Flower-detector-ai/
│
├── README.md
├── .gitignore
│
├── notebook/
│   └── advanced_training.ipynb
│
└── web/
    ├── index.html
    ├── flower_model.onnx    <-- (Model output goes here)
    ├── labels.json          <-- (Class mappings go here)
    └── static/
        ├── css/
        └── js/
```

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/anshrajsingh-dev/Flower-detector-ai.git
```

### Open the project

```bash
cd Flower-detector-ai
```

### Start a local server

```bash
python -m http.server 8000
```

### Open in your browser

```
http://localhost:8000
```

Upload a flower image and let the AI predict the flower species.

---

## 📂 Dataset

The training dataset is **not included** in this repository to keep the project lightweight.

---

## 💡 Future Improvements

* 🚀 Expand dataset to support 30+ flower species.
* 📈 Fine-tune deeper layers of the MobileNet backbone to push past the current 91.67% test accuracy.

---

## 👨‍💻 Author

**Ansh Raj Singh**

GitHub: https://github.com/anshrajsingh-dev

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐.
