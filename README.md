# 🧠 Sign Language Recognition using TensorFlow, SVM, and Random Forest

This project aims to develop an **AI-based sign language recognition system** by comparing multiple machine learning models.  
Using extracted feature data, three models were trained and evaluated: **RandomForest**, **SVM**, and a **TensorFlow-based Artificial Neural Network (ANN)**.  
The project also includes **real-time sign recognition** through webcam input.

---

## 📊 Model Comparison Results

| Model | Accuracy | Training Time |
|--------|-----------|----------------|
| RandomForest | 0.9345 | 5.53s |
| SVM | 0.9259 | 17.62s |
| **TensorFlow_ANN** | **0.9389** | **87.77s** |

✅ **Best Model:** TensorFlow_ANN  
✅ Model saved as: `best_sign_language_model.h5`  
✅ Label Encoder saved as: `label_encoder.joblib`

---

## 🧩 TensorFlow_ANN Model Performance

The model performs highly across **29 classes** (A–Z, *del*, *space*).  
**Overall accuracy:** 93.89%

**Macro average F1-score:** 0.93  
**Weighted average F1-score:** 0.94  

### Best Performing Classes
- **C**, **F**, **H**, **L**, **Z**, **space**

### Classes Requiring Improvement
- **M**, **N**, **R**, **U**

---

## ⚙️ Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **OpenCV**
- **MediaPipe**
- **Joblib**
- **NumPy**, **Pandas**, **Matplotlib**

---

## 🚀 Usage

### 1️⃣ Load the Model
```python
from tensorflow.keras.models import load_model
import joblib

model = load_model('best_sign_language_model.h5')
label_encoder = joblib.load('label_encoder.joblib')
```

### 2️⃣ Make Predictions
```python
prediction = model.predict(new_data)
predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]
print("Predicted sign:", predicted_label)
```

---

## 🎥 Real-Time Detection (canli.ipynb)

The project supports **real-time sign language recognition** using webcam input.  
You can run the live detection notebook to test it in action.

### Run:
```bash
jupyter notebook canli.ipynb
```

### Requirements:
- TensorFlow  
- OpenCV  
- MediaPipe  
- Joblib  
- NumPy  

---

## 📈 Future Work
- Data augmentation for underperforming classes  
- Deeper neural architectures for improved accuracy  
- Integration with mobile or web applications  

---

## 👨‍💻 Developed by
**Developed by [osmanvarisli](https://github.com/osmanvarisli)**

---

## ⚠️ License

**© 2025 osmanvarisli. All rights reserved.**

This project is released for **educational and research viewing purposes only**.  
You are **not allowed** to:
- Copy, modify, or redistribute the source code  
- Use it for commercial or derivative projects  

You **may**:
- View and study the source code for personal learning and research  
