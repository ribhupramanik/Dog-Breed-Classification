
# 🐶 Dog Breed Classifier – Flask Web App

This project is a **Dog Breed Classification Web Application** built using **Flask**, **TensorFlow/Keras**, and **MobileNet**.  
Users can upload an image of a dog and receive predictions for the most likely dog breeds.

---

## 🚀 Features

- Upload an image of a dog and get top-3 breed predictions  
- Backend built using **Flask**  
- Deep learning model trained on TensorFlow  
- Uses transfer learning (MobileNet)  
- Clean UI for uploading and displaying results  
- Labels and class names loaded from CSV  
- Data preprocessing identical to the original Jupyter Notebook  
- Fully ready for deployment

---

## 📁 Project Structure

```
/project-root
│── app.py
│── utils.py
│── requirements.txt
│── README.md
│
├── /model
│    ├── dog_breed_model.h5
│    ├── labels.csv
│
├── /static
│    └── /uploads    ← uploaded images stored here
│
└── /templates
     ├── index.html
     └── result.html
```

---

## 🧠 Model

The model is built using:

- **MobileNet + KerasLayer (TensorFlow Hub)**
- Preprocessing:
  - Decode JPEG
  - Convert to float32
  - Resize to 224×224
  - Normalize to [0,1]
- Labels loaded using `np.unique()` to match training order

---

## ⚙ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/dog-breed-classifier.git
cd dog-breed-classifier
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
venv\Scriptsctivate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Run the App

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 🖼 Uploading an Image

- Go to the homepage  
- Select an image (`.jpg`, `.jpeg`, `.png`)  
- Click **Upload & Predict**  
- View the prediction results and confidence scores  

---

## 📦 Requirements (requirements.txt)

```
Flask
numpy
tensorflow
tensorflow_hub
pandas
tf-keras
```

---

## 🧪 Preprocessing (utils.py)

```python
def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    return image
```

---

## 📌 Important Notes

- Do **not** reorder or modify label CSV file
- Model must match the preprocessing pipeline
- `CLASS_NAMES = np.unique(labels_df['breed'])` ensures correct mapping
- Ensure you are loading the exact same model used during training

---

## 📝 License

This project is open-source and free to use.  
Feel free to modify, improve, and share!

---

## ❤️ Contributing

Pull requests are welcome!  
If you’d like to contribute:
- Improve UI  
- Add more dog breeds  
- Enhance model accuracy  
- Add Docker deployment

---

## ⭐ Support

If you like this project, please give the **repository a star** on GitHub!

---

Enjoy building! 🐾
