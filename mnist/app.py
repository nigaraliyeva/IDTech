from flask import Flask, request, jsonify, render_template_string
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('emnist_model.h5')
print("Model loaded!")


def get_emnist_mapping():
    mapping = {}
    for i in range(10):
        mapping[i] = str(i)
    for i in range(26):
        mapping[10 + i] = chr(ord('A') + i)
    for i in range(26):
        mapping[36 + i] = chr(ord('a') + i)
    return mapping

label_map = get_emnist_mapping()

def preprocess_image(image_data):
    """Convert canvas image to 28x28"""

    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))


    image = image.convert('L')


    image = image.resize((28, 28), Image.Resampling.LANCZOS)


    img_array = np.array(image)


    img_array = 255 - img_array
    img_array = np.flip(img_array, axis=0)


    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMNIST Character Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }

        .canvas-container {
            border: 3px solid #667eea;
            border-radius: 10px;
            margin-bottom: 20px;
            background: white;
            cursor: crosshair;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        #canvas {
            display: block;
            touch-action: none;
        }

        .buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }

        button {
            flex: 1;
            padding: 15px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }

        #predictBtn {
            background: #667eea;
            color: white;
        }

        #predictBtn:hover {
            background: #5568d3;
            transform: translateY(-2px);
        }

        #clearBtn {
            background: #f093fb;
            color: white;
        }

        #clearBtn:hover {
            background: #d97ae8;
            transform: translateY(-2px);
        }

        .results {
            display: none;
        }

        .results.show {
            display: block;
        }

        .results h3 {
            margin-bottom: 15px;
            color: #333;
        }

        .result-item {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .character {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
            min-width: 50px;
        }

        .confidence-bar {
            flex: 1;
            margin: 0 20px;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.5s;
        }

        .confidence-text {
            font-weight: bold;
            color: #333;
            min-width: 60px;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✍️ EMNIST Character Recognition</h1>
        <p class="subtitle">Handwritten Digit & Letter Recognizer</p>

        <div class="canvas-container">
            <canvas id="canvas" width="280" height="280"></canvas>
        </div>

        <div class="buttons">
            <button id="clearBtn">🗑️ Clear</button>
            <button id="predictBtn">🔍 Predict</button>
        </div>

        <div id="results" class="results">
            <h3>Top 3 Predictions:</h3>
            <div id="predictions"></div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const clearBtn = document.getElementById('clearBtn');
        const predictBtn = document.getElementById('predictBtn');
        const resultsDiv = document.getElementById('results');
        const predictionsDiv = document.getElementById('predictions');

        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#000';

        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            resultsDiv.classList.remove('show');
        }

        clearCanvas();

        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            [lastX, lastY] = [e.offsetX, e.offsetY];
        });

        canvas.addEventListener('mouseup', () => isDrawing = false);
        canvas.addEventListener('mouseout', () => isDrawing = false);

        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            isDrawing = true;
            [lastX, lastY] = [touch.clientX - rect.left, touch.clientY - rect.top];
        });

        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (!isDrawing) return;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            [lastX, lastY] = [x, y];
        });

        canvas.addEventListener('touchend', () => isDrawing = false);

        clearBtn.addEventListener('click', clearCanvas);

        predictBtn.addEventListener('click', async () => {
            const imageData = canvas.toDataURL('image/png');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: imageData })
                });

                const data = await response.json();

                if (data.success) {
                    displayResults(data.predictions);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Connection error: ' + error);
            }
        });

        function displayResults(predictions) {
            predictionsDiv.innerHTML = '';

            predictions.forEach((pred, index) => {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.style.animationDelay = `${index * 0.1}s`;

                const confidence = (pred.confidence * 100).toFixed(1);

                item.innerHTML = `
                    <div class="character">${pred.character}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                    <div class="confidence-text">${confidence}%</div>
                `;

                predictionsDiv.appendChild(item);
            });

            resultsDiv.classList.add('show');
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']


        processed_image = preprocess_image(image_data)


        predictions = model.predict(processed_image, verbose=0)[0]


        top_3_indices = np.argsort(predictions)[-3:][::-1]

        results = []
        for idx in top_3_indices:
            character = label_map.get(idx, f"?{idx}")
            confidence = float(predictions[idx])
            results.append({
                'character': character,
                'confidence': confidence
            })

        return jsonify({'success': True, 'predictions': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
