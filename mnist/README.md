# ✍️ EMNIST Character Recognizer

A web application that recognizes handwritten characters (digits and letters) using a CNN model trained on the EMNIST dataset.

## Features
- Recognizes digits (0-9) and letters (A-Z, a-z) — 62 classes total
- Interactive drawing canvas
- Top 3 predictions with confidence scores
- Flask backend + vanilla JS frontend

## Dataset
[EMNIST Dataset](https://www.kaggle.com/datasets/crawford/emnist) — ByClass split  
100,000 train / 20,000 test samples used

## Model Architecture
- 3x Conv2D layers with BatchNormalization
- MaxPooling + Dropout (0.5)
- Dense(256) + Softmax(62)

## Project Structure
emnist-recognizer/
├── app.py # Flask backend
├── mnist_train.ipynb # Model training notebook
├── mnist_deploy.ipynb # Deployment notebook
├── emnist_model.h5 # Trained model
├── label_mapping.csv # Label to character mapping
├── requirements.txt # Dependencies
└── README.md

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/SƏNIN_USERNAME/emnist-recognizer.git
   cd emnist-recognizer
2. Install dependencies
   pip install -r requirements.txt
3. Run the application
   python app.py
4. Open your browser and go to http://localhost:5000

 ## Tech Stack

- **Model:** TensorFlow / Keras (CNN)
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** EMNIST ByClass
