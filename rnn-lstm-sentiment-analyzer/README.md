# 🎬 RNN vs LSTM Sentiment Analyzer



A web application that compares SimpleRNN and LSTM deep learning models on IMDB movie review sentiment analysis in real-time.



---



## 📌 Project Overview



Users type a movie review and the app instantly shows how both models perceive the sentiment — side by side. This project highlights the key differences between SimpleRNN and LSTM architectures in handling sequential text data.



---



## 🧠 Models



| Model | Architecture | Test Accuracy |

|---|---|---|

| SimpleRNN | Embedding(10000, 128) → SimpleRNN(64) → Dense(1) | ~72% |

| LSTM | Embedding(10000, 128) → LSTM(64) → Dense(1) | ~86% |



Both models are trained on the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) using Keras.



---



 📁 Project Structure



rnn-lstm-sentiment-analyzer/

│

├── notebooks/

│ ├── train\_models.ipynb # Model training and evaluation

│ └── deploy\_app.ipynb # Flask server and frontend deployment

│

├── app/

│ └── index.html # Frontend UI

│

├── requirements.txt

└── README.md





---



## 🚀 How to Run



### Step 1 — Train the Models

1. Open `notebooks/train\_models.ipynb` in Google Colab

2. Run all cells sequentially

3. Models will be saved as `simple\_rnn\_model.h5` and `lstm\_model.h5`



### Step 2 — Deploy the App

1. Open `notebooks/deploy\_app.ipynb` in Google Colab

2. Upload the saved `.h5` model files

3. Run all cells sequentially

4. Copy the ngrok URL printed in the output

5. Open the URL in your browser



---



## 🖥️ App Features



- Text area for entering a movie review

- "Analyze Sentiment" button

- Side-by-side result cards for SimpleRNN and LSTM

- Score displayed as percentage

- Positive / Negative label with color coding



---



## 🔍 Key Finding



LSTM significantly outperforms SimpleRNN on context-heavy sentences.



**Example:** *"Great acting from the lead but the script was terrible and the plot had too many holes."*



| Model | Score | Label |

|---|---|---|

| SimpleRNN | 37.9% | 😞 Negative |

| LSTM | 9.5% | 😞 Negative |



SimpleRNN is confused by the positive word *"great"* at the beginning, while LSTM correctly understands the overall negative sentiment of the sentence.



---



## 🛠️ Tech Stack



- Python

- TensorFlow / Keras

- Flask + Flask-CORS

- pyngrok

- HTML / CSS / JavaScript



---



## ⚠️ Note



Model files (`.h5`) are not included in this repository due to file size limitations.

Run `train\_models.ipynb` to generate them.

## Demo

[Video Demo](https://drive.google.com/file/d/1Eh2ackhvUgbcTX-N11YuQDvDxR-IWqGp/view?usp=sharing)




