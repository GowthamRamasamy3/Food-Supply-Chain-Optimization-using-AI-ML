# Food-Supply-Chain-Optimization-using-AI-ML


**Smart Forecasting: AI in Food Supply Chain Optimization**

---

## **📌 Project Objective:**

To develop a robust, data-driven forecasting system to accurately predict the **demand for food products** (fruits and vegetables) in the supply chain by leveraging various **Machine Learning (ML)**, **Deep Learning (DL)**, and **Hybrid models**, thereby minimizing waste, optimizing inventory, and improving overall supply chain efficiency.

---

## **💡 Motivation:**

The food supply chain suffers from demand-supply mismatches due to inaccurate forecasts, seasonal and climatic variations, and lack of consideration for external factors (like festivals, holidays, and weather). Our project aims to bridge this gap using **AI and advanced forecasting techniques** to assist farmers, retailers, and vendors in making data-driven decisions.

---

## **🗂️ Dataset:**

* **Source:** Open-source "Food Demand Forecasting Dataset" (you can link this here if available).
* **Features:**

  * Historical sales data
  * Product information
  * Regional factors
  * Climate data (temperature, rainfall)
  * Holiday & festival indicators
* **Data Preprocessing:**

  * Handling missing values
  * Feature scaling & encoding
  * Time-series formatting
  * Incorporating external variables like climate and holiday data

---

## **⚙️ Methodology:**

### **1. Exploratory Data Analysis (EDA):**

* Trend & Seasonality detection using line plots and decomposition.
* Correlation analysis between sales, climate, and external factors.
* Outlier detection and treatment.

---

### **2. Model Development:**

#### **Machine Learning Models:**

* **Linear Regression**
* **Random Forest Regressor**
* **Decision Tree Regressor**
* **XGBoost Regressor**
* **LightGBM Regressor**
* **CatBoost Regressor**

#### **Deep Learning Models:**

* **LSTM (Long Short-Term Memory)**
* **Bi-LSTM (Bidirectional LSTM)**

#### **Statistical Models:**

* **ARIMA (Auto-Regressive Integrated Moving Average)**
* **SARIMA (Seasonal ARIMA)**

#### **Hybrid Model (Best Performing):**

* **LSTM + XGBoost Hybrid Model**

  * LSTM captures time-series patterns (long-term dependencies).
  * XGBoost handles feature importance, non-linearity, and residual corrections.
  * **Achieved 93.47% accuracy** – highest among all models tested.

---

### **3. Model Evaluation Metrics:**

* **RMSE (Root Mean Square Error)**
* **MAE (Mean Absolute Error)**
* **R² Score**
* **MAPE (Mean Absolute Percentage Error)**

> Hybrid LSTM + XGBoost showed **lowest RMSE and MAE**, indicating superior forecasting capability.

---

## **📈 Results & Observations:**

| Model                     | RMSE      | MAE       | R² Score   |
| ------------------------- | --------- | --------- | ---------- |
| Random Forest             | 210.5     | 180.4     | 0.87       |
| XGBoost                   | 190.3     | 170.1     | 0.89       |
| LSTM                      | 170.8     | 150.5     | 0.91       |
| **Hybrid LSTM + XGBoost** | **120.4** | **100.6** | **0.9347** |

* **Hybrid model performed best** due to its ability to model both temporal dependencies and complex feature interactions.
* Seasonal & holiday data improved performance significantly.

---

## **🛠️ Technology Stack:**

* **Languages:** Python
* **Libraries/Frameworks:**

  * Pandas, NumPy, Scikit-Learn
  * XGBoost, LightGBM, CatBoost
  * TensorFlow, Keras (for LSTM)
  * Statsmodels (for ARIMA)
  * Matplotlib, Seaborn (for visualization)
* **Environment:** Jupyter Notebook, VS Code

---

## **📂 Suggested GitHub Repo Structure:**

```
Smart-Forecasting-Food-Supply-Chain/
│
├── data/                # Raw and processed datasets (CSV, Excel)
│
├── notebooks/           # Jupyter Notebooks for EDA, modeling, and evaluation
│   ├── 01_EDA.ipynb
│   ├── 02_ML_Models.ipynb
│   ├── 03_DL_Models.ipynb
│   ├── 04_Hybrid_Model.ipynb
│
├── models/              # Saved models (pkl, h5 files)
│   ├── rf_model.pkl
│   ├── lstm_model.h5
│   ├── hybrid_model.pkl
│
├── reports/             # PDF or markdown reports
│   ├── Final_Report.pdf
│   ├── Summary.md
│
├── src/                 # Python scripts for preprocessing, modeling
│   ├── preprocess.py
│   ├── model_train.py
│   ├── evaluate.py
│
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── LICENSE
```

---

## **🔮 Future Work:**

* Deploy the model as an API using **FastAPI/Flask**.
* Real-time data integration using weather and market APIs.
* Automate inventory management systems using forecast outputs.
* Extend forecasting to **perishable dairy, meat, and seafood products**.

---

## **🤝 Contribution:**

Open to collaboration for enhancing model generalizability, adding new datasets, or integrating with supply chain ERP systems.

---

## **📃 License:**

MIT License / Apache 2.0 (as per your preference).

---

## **📌 Sample GitHub Repo Name Suggestions:**

* `Food-Supply-Chain-Demand-Forecasting-AI`
* `SmartFoodDemand-Prediction`
* `AI-Food-Supply-Chain-Optimizer`

---

Great question, Gowtham! Here’s a **sample `requirements.txt`** file for your project, considering the tools, libraries, and models you used in your **"Smart Forecasting: AI in Food Supply Chain Optimization"** project:

---

✅ **`requirements.txt` for Your Project:**

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
catboost==1.2.3
tensorflow==2.16.1
keras==3.3.3
statsmodels==0.14.2
matplotlib==3.9.0
seaborn==0.13.2
jupyter==1.0.0
notebook==7.2.0
scipy==1.13.0
joblib==1.4.2
```

---

**Explanation:**

| Library          | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| **pandas**       | Data manipulation, reading CSV/Excel              |
| **numpy**        | Numerical operations                              |
| **scikit-learn** | ML Models (Random Forest, Preprocessing, Metrics) |
| **xgboost**      | XGBoost Regression Model                          |
| **lightgbm**     | LightGBM Model                                    |
| **catboost**     | CatBoost Model                                    |
| **tensorflow**   | LSTM & Deep Learning Models                       |
| **keras**        | High-level neural network API                     |
| **statsmodels**  | Statistical models like ARIMA, SARIMA             |
| **matplotlib**   | Data Visualization                                |
| **seaborn**      | Statistical Data Visualization                    |
| **jupyter**      | Notebook environment                              |
| **notebook**     | Run Jupyter notebooks                             |
| **scipy**        | Scientific computing, used in statsmodels         |
| **joblib**       | Model serialization (saving/loading models)       |

---

⚠️ **Optional (for real-time deployment in the future):**

If you plan to deploy this model as an API:

```
flask==3.0.3
fastapi==0.111.0
uvicorn==0.29.0
```

---

## 🎯 **To install these requirements:**

```bash
pip install -r requirements.txt
```

---
Sure Gowtham! Below is a **simple and clear guide** on how anyone (even a beginner) can run your **Food Supply Chain Optimization project** from GitHub step by step. You can put this in your **`README.md` file** under the "How to Run" section.

---

## 🚀 **How to Run This Project: Smart Forecasting in Food Supply Chain Optimization**

---

### **1. Clone the Repository**

First, clone the GitHub repo to your local machine:

```bash
git clone https://github.com/your-username/Food-Supply-Chain-Forecasting.git
cd Food-Supply-Chain-Forecasting
```

---

### **2. Create a Virtual Environment (Recommended)**

```bash
python -m venv venv
```

Activate the environment:

* **Windows:**

  ```bash
  venv\Scripts\activate
  ```
* **Linux/Mac:**

  ```bash
  source venv/bin/activate
  ```

---

### **3. Install Required Libraries**

Install all required Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### **4. Prepare the Dataset**

Place the dataset files (`train.csv`, `test.csv`, or other data files) into the `/data` folder.

✅ Make sure:

* The dataset is preprocessed as per the **EDA notebook** or use `src/preprocess.py` to preprocess raw data.

---

### **6. View the Model Performance**

Each notebook will:

* Train the model(s)
* Display evaluation metrics (RMSE, MAE, R² Score)
* Plot graphs like:

  * Predicted vs Actual Demand
  * Residual Plots
  * Feature Importance (for tree models)

---

### **7. (Optional) Save and Load Models**

You can save trained models using:

```python
import joblib
joblib.dump(model, 'models/rf_model.pkl')
```

And load them later for inference:

```python
model = joblib.load('models/rf_model.pkl')
```

For LSTM (Keras):

```python
model.save('models/lstm_model.h5')
```

And to load:

```python
from keras.models import load_model
model = load_model('models/lstm_model.h5')
```

---


**Complete Code Execution In one Step :
After completion of installation of requirement.txt !**

```bash
streamlit run streamlit_app.py --server.address localhost --server.port 8501
```


### **8. (Optional) Run Python Scripts Instead of Notebooks**

If you want to run the project via Python scripts instead of notebooks:

```bash
python src/preprocess.py   # Preprocess data
python src/model_train.py  # Train models
python src/evaluate.py     # Evaluate models
```


## ❗ **Important Notes:**

* Make sure the dataset is inside the `data/` folder.
* Python version: **3.10+ recommended**
* GPU is optional (only needed for faster LSTM training).

---


