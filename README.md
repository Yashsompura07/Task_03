# Car Price Prediction with Machine Learning 🚗💰

A comprehensive machine learning project to predict car prices using multiple regression models and advanced data analysis techniques.

---

## 📋 Project Overview

Car price prediction is one of the major research areas in machine learning. The price of a car depends on numerous factors including:
- **Brand & Model Year** - Brand reputation and age
- **Engine Specifications** - Engine size and horsepower
- **Performance Metrics** - Fuel efficiency and mileage
- **Transmission Type** - Manual vs Automatic
- **Condition** - Overall vehicle condition

This project builds and compares multiple ML models to accurately predict car prices based on these features.

### Key Objectives:
- Build regression models for price prediction
- Compare model performance metrics
- Identify most important features affecting price
- Visualize model predictions vs actual values
- Analyze residuals and model errors
- Predict prices for new cars

---

## 📊 Dataset Information

**File:** `car_price_data.csv`

### Features:
| Feature | Description | Type |
|---------|-------------|------|
| Brand | Car brand/manufacturer | Categorical |
| Model_Year | Year of manufacture | Integer |
| Engine_Size | Engine displacement (liters) | Float |
| Horsepower | Engine horsepower (HP) | Integer |
| Mileage | Current mileage (km) | Integer |
| Fuel_Type | Petrol/Diesel/Hybrid | Categorical |
| Transmission | Manual/Automatic | Categorical |
| Seats | Number of seats | Integer |
| Condition | Excellent/Good/Fair | Categorical |
| **Price** | Car price (Target) | Integer |

### Dataset Statistics:
- **Total Records:** 90 cars
- **Features:** 9 (8 independent + 1 target)
- **Price Range:** $7,800 - $65,000
- **Average Price:** $15,850
- **Brands Included:** 20+ popular manufacturers
- **Time Period:** 2017-2020 model years

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Yashsompura07/car-price-prediction.git
cd car-price-prediction
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

**Install all at once:**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Complete Analysis

```bash
python car_price_prediction.py
```

This will:
1. Load and analyze the dataset
2. Preprocess the data
3. Train 3 different ML models
4. Compare model performance
5. Generate visualizations
6. Display predictions

### Example Code

```python
from car_price_prediction import CarPricePredictionModel

# Initialize model
model = CarPricePredictionModel('car_price_data.csv')

# Display dataset information
model.display_dataset_info()

# Analyze feature relationships
model.analyze_features()

# Preprocess data
model.preprocess_data()

# Train models
model.train_linear_regression()
model.train_random_forest()
model.train_gradient_boosting()

# Compare performance
comparison = model.compare_models()

# Generate visualizations
model.plot_actual_vs_predicted()
model.plot_residuals()
model.plot_feature_importance()

# Predict for a new car
new_car_features = {
    'Brand': 'Toyota',
    'Model_Year': 2020,
    'Engine_Size': 2.0,
    'Horsepower': 180,
    'Mileage': 15000,
    'Fuel_Type': 'Hybrid',
    'Transmission': 'Automatic',
    'Seats': 5,
    'Condition': 'Excellent'
}
predictions = model.predict_new_car(new_car_features)
```

---

## 🤖 Machine Learning Models

### 1. **Linear Regression**
- Simple linear model assuming linear relationship
- Baseline model for comparison
- Interpretable and fast
- **Best for:** Understanding feature relationships

### 2. **Random Forest**
- Ensemble of decision trees
- Handles non-linear relationships well
- Provides feature importance scores
- **Best for:** Balanced performance and interpretability

### 3. **Gradient Boosting**
- Sequential ensemble of weak learners
- Often provides best predictions
- Captures complex patterns
- **Best for:** Maximum predictive accuracy

---

## 📈 Model Performance Comparison

### Key Metrics:

| Metric | Description | Formula |
|--------|-------------|---------|
| **R² Score** | Coefficient of determination | Higher is better (0-1) |
| **RMSE** | Root Mean Squared Error | Lower is better |
| **MAE** | Mean Absolute Error | Average absolute difference |

### Typical Results (from our dataset):
```
Model                  R² Score    RMSE        MAE
─────────────────────────────────────────────────
Random Forest          0.8950      $2,145      $1,825
Gradient Boosting      0.8820      $2,295      $1,920
Linear Regression      0.7650      $3,780      $3,125
```

---

## 📊 Visualizations Generated

### 1. **Actual vs Predicted Prices**
- Scatter plot comparing model predictions with actual prices
- Perfect prediction line for reference
- Shows prediction accuracy for each model
- Generated: `actual_vs_predicted.png`

### 2. **Residual Analysis**
- Scatter plot of residuals (Actual - Predicted)
- Shows if model has systematic errors
- Ideal: randomly scattered around zero
- Generated: `residuals.png`

### 3. **Feature Importance**
- Bar chart of feature importance from Random Forest
- Shows which features most affect price
- Helps understand price drivers
- Generated: `feature_importance.png`

---

## 📁 Project Structure

```
car-price-prediction/
│
├── car_price_prediction.py       # Main ML script
├── car_price_data.csv            # Dataset
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── SETUP.md                      # Git setup guide
│
├── outputs/                      # Generated files
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   └── feature_importance.png
│
└── .gitignore
```

---

## 🔍 How the Model Works

### Step 1: Data Loading
```python
df = pd.read_csv('car_price_data.csv')
```

### Step 2: Exploratory Data Analysis (EDA)
- Analyze feature distributions
- Check for correlations with price
- Identify outliers and missing values

### Step 3: Data Preprocessing
- Handle missing values
- Encode categorical variables
- Split into training (80%) and testing (20%)
- Scale features using StandardScaler

### Step 4: Model Training
Each model learns from training data:
```python
model.fit(X_train, y_train)
```

### Step 5: Evaluation
Models evaluated on test set:
```python
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
```

### Step 6: Prediction
For new cars:
```python
new_price = model.predict(new_car_features)
```

---

## 💡 Key Insights from Data

### Top Price Drivers:
1. **Horsepower** - Strong positive correlation
2. **Engine Size** - Larger engines cost more
3. **Brand** - Premium brands cost significantly more
4. **Model Year** - Newer cars are more expensive
5. **Transmission** - Automatics typically cost more

### Price Ranges by Condition:
- **Excellent:** Average +15% premium
- **Good:** Average baseline
- **Fair:** Average -20% discount

### Depreciation Pattern:
- 2017 models: Average $11,500
- 2018 models: Average $13,200
- 2019 models: Average $14,500
- 2020 models: Average $16,800

---

## 🛠️ Customization

### Change Train-Test Split:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42  # 75-25 split
)
```

### Adjust Model Hyperparameters:
```python
# Random Forest - more trees
model = RandomForestRegressor(n_estimators=200, max_depth=15)

# Gradient Boosting - slower learning
model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=200)
```

### Add New Features:
```python
df['Age'] = 2024 - df['Model_Year']
df['Price_Per_HP'] = df['Price'] / df['Horsepower']
```

---

## 📚 Understanding the Output

### Console Output Example:
```
=======================================================================
CAR PRICE PREDICTION - DATASET OVERVIEW
=======================================================================

Dataset Shape: (90, 10)

Statistical Summary:
       Model_Year  Engine_Size  Horsepower    Mileage      Price
count       90.00       90.00       90.00       90.00      90.00
mean      2018.33        1.70      140.89    32244.44   15849.89
std          1.04        0.71       66.98    16012.41    10850.44
min       2017.00        1.00       65.00     8000.00    7800.00
max       2020.00        3.50      380.00    62000.00   65000.00
```

### Model Performance Example:
```
Training R² Score: 0.9124
Testing R² Score: 0.8950
Training RMSE: $1,895
Testing RMSE: $2,145
Testing MAE: $1,825
```

---

## 🔮 Making Predictions

### For a Specific Car:
```python
# Example: 2020 Toyota with high specs
new_car = {
    'Brand': 'Toyota',
    'Model_Year': 2020,
    'Engine_Size': 2.5,
    'Horsepower': 200,
    'Mileage': 20000,
    'Fuel_Type': 'Hybrid',
    'Transmission': 'Automatic',
    'Seats': 5,
    'Condition': 'Excellent'
}

predictions = model.predict_new_car(new_car)
# Output:
# Linear Regression: $28,500
# Random Forest: $29,200
# Gradient Boosting: $29,800
# Average Prediction: $29,167
```

---

## 🎯 Common Use Cases

1. **Car Dealerships:** Price used cars automatically
2. **Insurance Companies:** Assess vehicle value for claims
3. **Buyers:** Understand fair market prices
4. **Sellers:** Price cars competitively
5. **Traders:** Identify underpriced vehicles

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add more models (XGBoost, Neural Networks)
- Enhance features (location, color, trim level)
- Improve visualizations
- Add cross-validation
- Create price prediction app

### Steps:
1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

---

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Guide](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

---

## 📝 License

MIT License - feel free to use for education and projects

---

## 👤 Author

**Yash Sompura**
- GitHub: https://github.com/Yashsompura07

---

## 🙏 Acknowledgments

- Dataset inspired by real-world car pricing
- Scikit-learn library for ML algorithms
- Visualization libraries: Matplotlib & Seaborn

---

## ⚠️ Important Notes

1. **Data Quality:** Results depend on data quality and representativeness
2. **Feature Engineering:** More features often improve predictions
3. **Hyperparameter Tuning:** Results can vary with different parameters
4. **Cross-Validation:** Use for more robust evaluation
5. **Deployment:** Scale features consistently in production

---

## 🎓 Educational Value

This project teaches:
- ✅ Data loading and exploration
- ✅ Data preprocessing and scaling
- ✅ Building multiple ML models
- ✅ Model evaluation and comparison
- ✅ Visualization of results
- ✅ Making predictions on new data
- ✅ Feature importance analysis
- ✅ Residual analysis

---

**Last Updated:** June 2026
**Status:** Active ✅

Made with ❤️ for ML enthusiasts
