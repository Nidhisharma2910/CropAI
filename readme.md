Harvestify is an AI-powered agricultural assistant that helps farmers and agronomists make smarter decisions. It provides three core features: crop recommendation, fertilizer suggestion, and plant disease detection — all from a simple web interface.

🚀 Live Demo
👉 https://cropai-5.onrender.com/

✨ Features
🌱 Crop Recommendation
Input your soil's nitrogen, phosphorous, potassium, pH level, and rainfall. The app auto-fetches real-time temperature and humidity from your location using the Open-Meteo API (no API key needed) and recommends the best crop to grow using a trained Random Forest model.
🧪 Fertilizer Suggestion
Enter your soil nutrient levels and the crop you're growing. The app compares your values against ideal nutrient ratios and recommends exactly which fertilizer to apply and how.
🔬 Plant Disease Detection
Upload a photo of a plant leaf. A ResNet9 deep learning model analyzes the image and identifies the disease (if any) from 38 possible classes across 14 crop types, along with treatment recommendations.

🛠️ Tech Stack
LayerTechnologyBackendPython, FlaskML Model (Crop)Scikit-learn Random ForestML Model (Disease)PyTorch ResNet9 (CNN)Weather APIOpen-Meteo (free, no key needed)FrontendHTML, CSS, JavaScriptDeploymentRender

📁 Project Structure
Harvestify/
├── app/
│   ├── static/          # CSS, JS, images
│   ├── templates/        # HTML templates
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── disease.py    # Disease info dictionary
│   │   ├── fertilizer.py # Fertilizer info dictionary
│   │   └── model.py      # ResNet9 model definition
│   └── Data/
│       └── fertilizer.csv
├── models/
│   ├── plant_disease_model.pth   # Trained CNN weights
│   └── RandomForest.pkl          # Trained crop model
├── main.py              # Flask app entry point
├── requirements.txt
├── runtime.txt
└── README.md

⚙️ Local Setup
1. Clone the repository
bashgit clone https://github.com/Nidhisharma2910/CropAI

2. Create a virtual environment
bashpython -m venv cropenv
source cropenv/bin/activate        # Linux / Mac
cropenv\Scripts\activate           # Windows
3. Install dependencies
bashpip install -r requirements.txt
4. Run the app
bashpython main.py
Visit http://127.0.0.1:5000 in your browser.




🌿 Supported Crops — Disease Detection
CropDiseases DetectedAppleApple Scab, Black Rot, Cedar Apple RustCornCercospora Leaf Spot, Common Rust, Northern Leaf BlightGrapeBlack Rot, Esca, Leaf BlightTomatoBacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic VirusPotatoEarly Blight, Late BlightPeachBacterial SpotPepperBacterial SpotStrawberryLeaf ScorchAnd more...Cherry, Blueberry, Raspberry, Soybean, Squash, Orange

📊 Model Details
Crop Recommendation Model

Algorithm: Random Forest Classifier
Input Features: N, P, K, Temperature, Humidity, pH, Rainfall
Output: Recommended crop name

Plant Disease Detection Model

Architecture: ResNet9 (custom lightweight CNN)
Input: RGB leaf image (256×256)
Output: Disease class from 38 categories
Framework: PyTorch




