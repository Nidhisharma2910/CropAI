from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
from app.utils.disease import disease_dic
from app.utils.fertilizer import fertilizer_dic
from app.utils.model import ResNet9
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
import os
import traceback

# ─────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join('app', 'templates'),
    static_folder=os.path.join('app', 'static')
)

# ─────────────────────────────────────────────────────────────
# Disease Model
# ─────────────────────────────────────────────────────────────
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# ─────────────────────────────────────────────────────────────
# Crop Model
# ─────────────────────────────────────────────────────────────
crop_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_model_path, 'rb'))

# ─────────────────────────────────────────────────────────────
# Location Coordinates
# Covers every state/UT in your dropdown + major cities
# ─────────────────────────────────────────────────────────────
LOCATION_COORDS = {
    # ── States / UTs (mapped to capital city coords) ──────────
    "andaman & nicobar":        (11.7401,  92.6586),
    "andaman and nicobar":      (11.7401,  92.6586),
    "andhra pradesh":           (15.9129,  79.7400),
    "arunachal pradesh":        (27.0844,  93.6053),
    "assam":                    (26.1445,  91.7362),
    "bihar":                    (25.5941,  85.1376),
    "chandigarh":               (30.7333,  76.7794),
    "chhattisgarh":             (21.2514,  81.6296),
    "dadra & nagar haveli":     (20.1809,  73.0169),
    "dadra and nagar haveli":   (20.1809,  73.0169),
    "daman & diu":              (20.3974,  72.8328),
    "daman and diu":            (20.3974,  72.8328),
    "delhi":                    (28.6139,  77.2090),
    "new delhi":                (28.6139,  77.2090),
    "goa":                      (15.2993,  74.1240),
    "gujarat":                  (23.2156,  72.6369),
    "haryana":                  (29.0588,  76.0856),
    "himachal pradesh":         (31.1048,  77.1734),
    "jammu & kashmir":          (34.0837,  74.7973),
    "jammu and kashmir":        (34.0837,  74.7973),
    "jharkhand":                (23.3441,  85.3096),
    "karnataka":                (12.9716,  77.5946),
    "kerala":                   (8.5241,   76.9366),
    "lakshadweep":              (10.5667,  72.6417),
    "madhya pradesh":           (23.2599,  77.4126),
    "maharashtra":              (19.0760,  72.8777),
    "manipur":                  (24.8170,  93.9368),
    "meghalaya":                (25.5788,  91.8933),
    "mizoram":                  (23.1645,  92.9376),
    "nagaland":                 (25.6701,  94.1077),
    "orissa":                   (20.2961,  85.8245),
    "odisha":                   (20.2961,  85.8245),
    "pondicherry":              (11.9416,  79.8083),
    "puducherry":               (11.9416,  79.8083),
    "punjab":                   (30.7333,  76.7794),
    "rajasthan":                (26.9124,  75.7873),
    "sikkim":                   (27.3314,  88.6138),
    "tamil nadu":               (13.0827,  80.2707),
    "tripura":                  (23.8315,  91.2868),
    "uttar pradesh":            (26.8467,  80.9462),
    "uttaranchal":              (30.3165,  78.0322),
    "uttarakhand":              (30.3165,  78.0322),
    "west bengal":              (22.5726,  88.3639),

    # ── Major Cities ──────────────────────────────────────────
    "mumbai":                   (19.0760,  72.8777),
    "pune":                     (18.5204,  73.8567),
    "bangalore":                (12.9716,  77.5946),
    "bengaluru":                (12.9716,  77.5946),
    "hyderabad":                (17.3850,  78.4867),
    "chennai":                  (13.0827,  80.2707),
    "kolkata":                  (22.5726,  88.3639),
    "jaipur":                   (26.9124,  75.7873),
    "lucknow":                  (26.8467,  80.9462),
    "patna":                    (25.5941,  85.1376),
    "bhopal":                   (23.2599,  77.4126),
    "ahmedabad":                (23.0225,  72.5714),
    "surat":                    (21.1702,  72.8311),
    "nagpur":                   (21.1458,  79.0882),
    "indore":                   (22.7196,  75.8577),
    "amritsar":                 (31.6340,  74.8723),
    "coimbatore":               (11.0168,  76.9558),
    "visakhapatnam":            (17.6868,  83.2185),
    "agra":                     (27.1767,  78.0081),
    "varanasi":                 (25.3176,  82.9739),
    "jodhpur":                  (26.2389,  73.0243),
    "thiruvananthapuram":       (8.5241,   76.9366),
    "mysuru":                   (12.2958,  76.6394),
    "mysore":                   (12.2958,  76.6394),
    "raipur":                   (21.2514,  81.6296),
    "ranchi":                   (23.3441,  85.3096),
    "guwahati":                 (26.1445,  91.7362),
    "bhubaneswar":              (20.2961,  85.8245),
    "shimla":                   (31.1048,  77.1734),
    "dehradun":                 (30.3165,  78.0322),
    "gandhinagar":              (23.2156,  72.6369),
    "panaji":                   (15.4909,  73.8278),
    "imphal":                   (24.8170,  93.9368),
    "shillong":                 (25.5788,  91.8933),
    "aizawl":                   (23.1645,  92.9376),
    "kohima":                   (25.6701,  94.1077),
    "agartala":                 (23.8315,  91.2868),
    "gangtok":                  (27.3314,  88.6138),
    "itanagar":                 (27.0844,  93.6053),
    "port blair":               (11.7401,  92.6586),
    "silvassa":                 (20.2766,  73.0075),
    "daman":                    (20.3974,  72.8328),
    "kavaratti":                (10.5667,  72.6417),
    "dispur":                   (26.1445,  91.7362),
    "amaravati":                (16.5062,  80.6480),
    "srinagar":                 (34.0837,  74.7973),
    "jammu":                    (32.7266,  74.8570),
}


def get_lat_lon(city_or_state):
    """Convert city/state name → (lat, lon). Local map first, then geocoding API."""
    if not city_or_state:
        print("❌ Empty city/state input")
        return None, None

    key = city_or_state.strip().lower()

    if key in LOCATION_COORDS:
        lat, lon = LOCATION_COORDS[key]
        print(f"📍 '{city_or_state}' → local map: lat={lat}, lon={lon}")
        return lat, lon

    try:
        url = (f"https://geocoding-api.open-meteo.com/v1/search"
               f"?name={city_or_state.strip()}&count=1&language=en&format=json")
        print(f"🌐 Geocoding API: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            lat = data["results"][0]["latitude"]
            lon = data["results"][0]["longitude"]
            print(f"📍 Geocoding API → lat={lat}, lon={lon}")
            return lat, lon
        else:
            print(f"❌ '{city_or_state}' not found. API response: {data}")
            return None, None

    except requests.exceptions.ConnectionError:
        print("❌ Geocoding: No network")
    except requests.exceptions.Timeout:
        print("❌ Geocoding: Timed out")
    except Exception as e:
        print(f"❌ Geocoding error: {e}")

    return None, None


# ─────────────────────────────────────────────────────────────
# Weather Fetch (Open-Meteo — completely free, no API key)
# ─────────────────────────────────────────────────────────────
def weather_fetch(lat, lon):
    """Returns (temperature, humidity) or (None, None) on failure."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=relative_humidity_2m"
        f"&forecast_days=1"
    )
    print(f"🌐 Weather API: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "current_weather" not in data:
            print(f"❌ 'current_weather' missing. Response: {data}")
            return None, None

        temperature = data["current_weather"]["temperature"]
        humidity    = data["hourly"]["relative_humidity_2m"][0]

        print(f"✅ Weather → {temperature}°C, {humidity}% humidity")
        return temperature, humidity

    except requests.exceptions.ConnectionError:
        print("❌ Weather: No network")
    except requests.exceptions.Timeout:
        print("❌ Weather: Timed out")
    except requests.exceptions.HTTPError as e:
        print(f"❌ Weather: HTTP error {e}")
    except KeyError as e:
        print(f"❌ Weather: Missing key {e}")
    except Exception as e:
        print(f"❌ Weather: {e}")

    return None, None


# ─────────────────────────────────────────────────────────────
# Disease Prediction
# ─────────────────────────────────────────────────────────────
def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', title='Harvestify - Home')


@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='Harvestify - Crop Recommendation')


@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='Harvestify - Fertilizer Suggestion')


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        try:
            # ── Step 1: Read form fields ──────────────────────────────
            N        = int(request.form['nitrogen'])
            P        = int(request.form['phosphorous'])
            K        = int(request.form['pottasium'])
            ph       = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            city     = request.form.get("city", "").strip()
            state    = request.form.get("state", "").strip()

            print(f"\n{'='*55}")
            print(f"📋 N={N} P={P} K={K} ph={ph} rainfall={rainfall}")
            print(f"   city='{city}'  state='{state}'")

            # ── Step 2: Get coordinates (city first, state as fallback)
            lat, lon = None, None

            if city:
                lat, lon = get_lat_lon(city)

            if lat is None and state:
                print(f"⚠️  City lookup failed — trying state '{state}'")
                lat, lon = get_lat_lon(state)

            if lat is None:
                print("❌ STOP: No coordinates found.")
                return render_template('try_again.html', title='Crop Recommendation')

            # ── Step 3: Fetch weather ─────────────────────────────────
            temperature, humidity = weather_fetch(lat, lon)

            if temperature is None or humidity is None:
                print("❌ STOP: Weather fetch failed.")
                return render_template('try_again.html', title='Crop Recommendation')

            # ── Step 4: Predict ───────────────────────────────────────
            # Order: N, P, K, temperature, humidity, ph, rainfall
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            print(f"🔢 Model input: {data}")

            prediction = crop_recommendation_model.predict(data)[0]
            print(f"🌾 Prediction: {prediction}")
            print(f"{'='*55}\n")

            return render_template('crop-result.html', prediction=prediction, title='Crop Recommendation')

        except KeyError as e:
            print(f"❌ Missing form field: {e}  ← Check name= in crop.html")
            traceback.print_exc()
            return render_template('try_again.html', title='Crop Recommendation')

        except ValueError as e:
            print(f"❌ Bad input value: {e}")
            traceback.print_exc()
            return render_template('try_again.html', title='Crop Recommendation')

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            traceback.print_exc()
            return render_template('try_again.html', title='Crop Recommendation')


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    try:
        crop_name = str(request.form['cropname'])
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        print(f"🌱 Fertilizer: crop={crop_name} N={N} P={P} K={K}")

        df = pd.read_csv('app/Data/fertilizer.csv')

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K

        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]

        if max_value == "N":
            key = 'NHigh' if n < 0 else 'Nlow'
        elif max_value == "P":
            key = 'PHigh' if p < 0 else 'Plow'
        else:
            key = 'KHigh' if k < 0 else 'Klow'

        print(f"🧪 Fertilizer key: {key}")
        response = Markup(fertilizer_dic[key])
        return render_template('fertilizer-result.html', recommendation=response, title='Fertilizer Suggestion')

    except Exception as e:
        print(f"❌ Fertilizer error: {e}")
        traceback.print_exc()
        return render_template('try_again.html', title='Fertilizer Suggestion')


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title='Disease Detection')
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(disease_dic[prediction])
            return render_template('disease-result.html', prediction=prediction, title='Disease Detection')
        except Exception as e:
            print(f"❌ Disease error: {e}")
            traceback.print_exc()
            return render_template('try_again.html', title='Error')
    return render_template('disease.html', title='Disease Detection')


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)