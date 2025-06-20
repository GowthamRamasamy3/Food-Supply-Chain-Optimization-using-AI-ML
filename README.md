# Food Supply Optimization Model

A comprehensive system for forecasting meal demand across multiple distribution centers and optimizing the food supply chain. This solution includes a Streamlit dashboard with interactive visualizations, machine learning models for demand forecasting, and email notification features.

## Features

- **Data Exploration**: Interactive visualization of historical demand data
- **Demand Forecasting**: Hybrid ML approach (LSTM + XGBoost) to predict future demand
- **Supply Chain Optimization**: Inventory planning with safety buffer and lead time considerations
- **Climate Impact Analysis**: Assessment of environmental impact with sustainability recommendations
- **Email Notifications**: Alert system for demand threshold monitoring

## Files and Components

- **streamlit_app.py**: Main Streamlit dashboard application
- **database.py**: PostgreSQL database connection and operations
- **email_service.py**: Email notification service
- **notification_scheduler.py**: Background scheduler for checking and sending notifications
- **test_notifications.py**: Script to test the notification system
- **model.py**: ML model implementation (LSTM + XGBoost)
- **data_processing.py**: Data preparation and preprocessing
- **climate_impact.py**: Environmental impact analysis

## Setup and Installation

1. Ensure Python 3.11+ is installed
2. Install required packages:
   ```
   pip install streamlit pandas numpy plotly matplotlib email-validator psycopg2-binary
   ```
3. Configure PostgreSQL database credentials in environment variables
4. Configure SMTP settings for email notifications:
   - SMTP_USERNAME
   - SMTP_PASSWORD
   - SMTP_SERVER (default: smtp.gmail.com)
   - SMTP_PORT (default: 587)

## Running the Application

Start the Streamlit dashboard:
```
streamlit run streamlit_app.py
```

## Usage

1. **Data Loading**: Select a base year and load historical data
2. **Model Training**: Configure model parameters and generate forecasts
3. **Supply Chain Optimization**: Set safety buffer, lead time, and cost parameters
4. **Climate Impact Analysis**: Examine environmental impact and get sustainability recommendations
5. **Email Notifications**: Register and set up demand threshold alerts

## Notification System

The application includes a background scheduler that checks demand thresholds and sends email notifications when conditions are met. To test the notification system, run:
```
python test_notifications.py
```

## Database Tables

- **users**: Store user information for notifications
- **notification_preferences**: User-defined alert thresholds
- **notifications**: History of sent notifications

---

## ✅ **`requirements.txt` for Your Project:**

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

## **Explanation:**

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

## ⚠️ **Optional (for real-time deployment in the future):**

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


### **6. View the Model Performance**

Each notebook will:

* Train the model(s)
* Display evaluation metrics (RMSE, MAE, R² Score)
* Plot graphs like:

  * Predicted vs Actual Demand
  * Residual Plots
  * Feature Importance (for tree models)

---

**EASY WAY TO RUN THE CODE WITH ONE STEP AFTER THE INSTALLATION OF requirement.txt**

```bash
streamlit run streamlit_app.py --server.address localhost --server.port 8501
```
**The above code run with Streamlit**


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

### **8. (Optional) Run Python Scripts Instead of Notebooks**

If you want to run the project via Python scripts instead of notebooks:

```bash
python src/preprocess.py   # Preprocess data
python src/model_train.py  # Train models
python src/evaluate.py     # Evaluate models
```

---

## ✅ **Summary of Commands:**

```bash
# Clone the repo
git clone https://github.com/your-username/Food-Supply-Chain-Forecasting.git
cd Food-Supply-Chain-Forecasting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## ❗ **Important Notes:**

* Make sure the dataset is inside the `data/` folder.
* Python version: **3.10+ recommended**
* GPU is optional (only needed for faster LSTM training).

---






