# 💓 Heart Disease Risk Predictor

A machine learning web application built with **Flask** and **HTML/CSS** that predicts the risk of heart disease based on user input. The model uses a combination of lifestyle, medical, and demographic features to make predictions in real time — with a clean interface and no page reloads!

---

## 🚀 Features

- 🔍 Logistic Regression model trained on real-world data
- 🖥️ Modern UI with a Spline animation background
- ⚡ Instant prediction with **AJAX** (no page reload)
- 📈 Real-time input validation
- 🧠 Handles both numeric and categorical inputs
- 📦 Modular structure with `joblib` model loading
- 🔐 Clean backend logic using Flask and scikit-learn

---

## 📊 Tech Stack

| Tech              | Description                           |
|-------------------|---------------------------------------|
| Python + Flask    | Backend and model inference logic     |
| HTML/CSS          | Frontend UI and animation integration |
| JavaScript (AJAX) | Dynamic form submission (no reload)   |
| scikit-learn      | Model training + preprocessing         |
| pandas, joblib    | Data handling and model serialization |

---

## 🧠 Model Inputs

The app takes in a range of medical, demographic, and lifestyle inputs:

- **Numerical**:  
  `age`, `blood_pressure`, `cholesterol`, `max_heart_rate`, `oldpeak`, `ca`

- **Categorical**:  
  `sex`, `smoking`, `diet_type`, `alcohol_intake`, `stress_level`,  
  `physical_activity_level`, `work_life_balance`, `family_history`,  
  `urban_or_rural`, `Diabetes`, `HadAngina`, `Low_HDL_Cholesterol`,  
  `High_LDL_Cholesterol`

> Inputs are preprocessed (scaling + encoding) and passed to a trained classifier.

---

## 🧪 Model Training Overview

This project uses a **Logistic Regression** model to predict whether an individual is at risk of heart disease based on lifestyle, biological, and medical history data.

- 📊 **Dataset Size**: 30,000 anonymized patient records
- 🧾 **Target**: Binary classification  
  `1 = Heart Disease Detected`, `0 = No Heart Disease`

### 🧹 Preprocessing Steps

- 🧼 Null/missing values handled appropriately  
- 🔢 Numerical features scaled using `StandardScaler`  
- 🏷️ Categorical features encoded using `OneHotEncoder`

### 🔍 Key Predictive Features

| Feature                    | Description                                      |
|----------------------------|--------------------------------------------------|
| `age`                      | Age in years                                     |
| `blood_pressure`           | Resting blood pressure (mm Hg)                   |
| `cholesterol`              | Serum cholesterol (mg/dl)                        |
| `max_heart_rate`           | Maximum heart rate achieved                      |
| `oldpeak`                  | ST depression induced by exercise                |
| `ca`                       | Number of major vessels colored by fluoroscopy   |
| `smoking`, `alcohol_intake`| Lifestyle risk factors                           |
| `diet_type`                | Eating habits (Vegetarian, Non-Veg, Vegan)       |
| `stress_level`             | Self-reported stress levels                      |
| `physical_activity_level`  | Sedentary → Active spectrum                      |
| `Diabetes`, `HadAngina`    | Medical history indicators                       |
| `family_history`, `urban_or_rural` | Socioeconomic/Genetic influence       |

The model was selected for its simplicity and good generalization on binary medical classification tasks.

---
