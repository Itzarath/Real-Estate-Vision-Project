# Multimodal Real Estate Price Predictor 🏠📸

**A machine learning pipeline that fuses traditional tabular data with satellite imagery to predict house prices with high precision.**

## 📖 Overview
Predicting real estate prices is often limited to numbers like square footage and bedroom count. However, visual factors—such as roof condition, neighborhood density, and proximity to water—play a massive role in valuation. 

This project implements a **Late Fusion Architecture** that combines:
1.  **Tabular Data:** Metadata (size, year built, location).
2.  **Visual Data:** High-resolution satellite imagery fetched via the **Mapbox API**.

By extracting visual embeddings using **EfficientNet-B0** and fusing them with tabular features in an **XGBoost** regressor, we achieved a significant performance improvement over baseline models.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `data_fetcher.py` | **Data Collection:** Script to download satellite images for every property using the Mapbox Static API. |
| `data_preprocessing_and_model_training.ipynb` | Creates geospatial features,EfficientNet feature Extraction, PCA Compression and 5-fold XGBoost Training |
| `Project_Report.pdf` | **Full Report:** detailed analysis of methodology, financial insights, and results. |

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas numpy torch torchvision xgboost optuna scikit-learn tqdm requests

```

### 2. Data Collection

To download the satellite images, you need a Mapbox API key.

1. Open `data_fetcher.py`.
2. Replace `YOUR_MAPBOX_API_KEY_HERE` with your actual key.
3. Run the script:
```bash
python data_fetcher.py

```



### 3. Data Processing and Model training

Run `data_preprocessing_and_model_training.ipynb` to:

* Load the raw train/test Excel files.
* Engineer geospatial features (e.g., neighborhood density).
* Visualize price distribution.
* Save the cleaned dataset as `processed_data.csv`.
* **Visualize:** Use Grad-CAM to see what the CNN "looks" at (e.g., driveways, pools).
* **Extract:** Pass images through EfficientNet-B0 to get 1280-dimensional embeddings.
* **Fuse:** Compress embeddings with PCA (to 50 dims) and merge with tabular data.
* **Train:** Run the 5-Fold Cross-Validation Ensemble.
* **Predict:** Generate `submission_final.csv`.

---

## 🧠 Methodology

### Architecture

We utilize a **Late Fusion** approach:

1. **Image Branch:** * Input: 224x224 Satellite Image.
* Model: **EfficientNet-B0** (Pre-trained on ImageNet).
* Output: 1280 feature vector  Reduced to 50 via PCA.


2. **Tabular Branch:**
* Input: Bedrooms, Bathrooms, Sqft, Lat/Long.
* Feature Engineering: Micro-neighborhood density bins.


3. **Fusion & Prediction:**
* Concatenate Tabular Features + 50 Visual PCA Components.
* Model: **XGBoost Regressor** (Optimized via Optuna).



### Explainability (Grad-CAM)

To ensure the model isn't learning noise, we used Grad-CAM to visualize attention maps.

* **Result:** The model focuses heavily on the building structure and immediate driveway, confirming it learns relevant property features rather than background noise.

---

## 📊 Results

| Model Strategy | R² Score (Log Space) |
| --- | --- |
| **Baseline (Tabular Only)** | ~0.8950 |
| **Multimodal Fusion (Tabular + Vision)** | **0.9112** |

*Adding visual context provided a clear boost in accuracy, particularly for high-variance properties where "curb appeal" matters most.*

---

## 👤 Author

* **Barathvel M**

```

```
