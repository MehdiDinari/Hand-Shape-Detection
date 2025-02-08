# Hand Shape Detector - Random Forest Classifier

This project is a **Sign Language Detection** system using **Random Forest Classifier** trained on feature-extracted hand gesture data. The goal is to recognize hand signs based on numerical feature representations.

## 📌 Features
- **Machine Learning Model**: Uses a **Random Forest Classifier** for hand gesture recognition.
- **Data Handling**: Processes and validates input data before training.
- **Model Training**: Trains a classifier and evaluates its accuracy.
- **Model Persistence**: Saves the trained model for future inference.

---

## 🚀 Installation & Setup

### **1. Clone the Repository**
```sh
git clone https://github.com/mehdidinari/hand-shape-detector.git
cd hand-shape-detector
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```sh
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### **3. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4. Prepare the Dataset**
Ensure you have a `data.pickle` file containing the training dataset.
The dictionary should have:
- `data`: List of feature vectors (each of size 42)
- `labels`: Corresponding labels for classification

---

## 🏋️ Training the Model
Run the following command to train the classifier:
```sh
python train_classifier.py
```
This script:
- Loads the dataset
- Preprocesses and validates feature vectors
- Splits the data into training and testing sets
- Trains a **Random Forest Classifier**
- Evaluates accuracy and saves the trained model as `model.p`

---

## 📊 Model Performance
After training, the script prints the accuracy of the model:
```sh
Accuracy: 95.12%
```

---

## 📄 Usage
Once trained, you can load and use the model for prediction:

```python
import pickle
import numpy as np

# Load the trained model
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

# Example input feature vector (must be of length 42)
sample_input = np.random.rand(42).reshape(1, -1)

# Make a prediction
prediction = model.predict(sample_input)
print(f'Predicted Sign: {prediction[0]}')
```

---

## 🛠 Technologies Used
- **Python** 🐍
- **Scikit-Learn** (Machine Learning)
- **NumPy** (Data Processing)
- **Pickle** (Model Saving & Loading)
- **mediapipe**
- **cv2**

---

## 📌 To-Do / Future Enhancements
✅ Improve feature extraction from images/videos 📷
✅ Experiment with deep learning models (CNNs, LSTMs) 🤖
✅ Build a real-time sign language detection system 🎥

---

## 👨‍💻 Contributing
Pull requests are welcome! Feel free to open an issue for any suggestions or improvements.

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 🌟 Acknowledgments
- Thanks to the OpenAI community for guidance and resources!
- Special thanks to Scikit-Learn for making machine learning so accessible.


