# diamond-price-prediction
Diamond Price Prediction using Machine Learning
# Diamond Price Prediction - Machine Learning Project

## 📌 Project Overview

This project uses Machine Learning regression models to predict diamond prices based on their physical and quality characteristics (carat weight, cut, color, clarity, etc.).

The goal is to build and compare multiple regression models to accurately predict diamond prices and understand which factors have the most influence on pricing.

---

## 📊 Dataset

- **Dataset**: Seaborn Built-in Diamond Dataset
- **Total Records**: 53,940 diamonds
- **Features**: 9 (carat, cut, color, clarity, depth, table, x, y, z)
- **Target Variable**: `price` (ranging from $326 to $18,823)
- **Missing Values**: None (clean dataset)

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| carat | Weight of diamond (0.2 to 5.01 carats) |
| cut | Quality of cut (Fair, Good, Very Good, Premium, Ideal) |
| color | Diamond color (J to D, where D is best) |
| clarity | How clear it is (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF) |
| depth | Total depth percentage |
| table | Width of top relative to widest point |
| x, y, z | Physical dimensions in millimeters |
| price | **Target: Price in USD** |

---

## 🔧 Technologies Used

- **Python 3**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `scikit-learn` - Machine Learning algorithms
  - `numpy` - Numerical computations
  - `seaborn` - Data loading and visualization
  - `matplotlib` - Chart creation

---

## 📈 Project Workflow

### 1. Data Exploration
- Loaded 53,940 diamond records
- Analyzed data types and distributions
- Confirmed no missing values

### 2. Data Preprocessing
- Encoded categorical variables (cut, color, clarity) to numerical values using LabelEncoder
- All text columns converted to numbers for ML model compatibility

### 3. Feature Engineering
- Separated features (X): 9 columns
- Separated target (y): price column
- No additional features created (dataset already well-structured)

### 4. Train-Test Split
- Training set: 43,152 diamonds (80%)
- Test set: 10,788 diamonds (20%)
- Random seed: 42 (for reproducibility)

### 5. Model Training & Comparison
Trained and compared 3 regression models:

| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | 0.8851 (88.51%) | $1,351.26 |
| Decision Tree | 0.9679 (96.79%) | $716.73 |
| **Random Forest** | **0.9816 (98.16%)** | **$540.63** ✓ |

**Winner**: Random Forest - explains 98.16% of price variation with average error of $540.63

### 6. Feature Importance Analysis
Using Random Forest, identified which factors matter most for diamond pricing:

| Feature | Importance |
|---------|-----------|
| carat | 43.2% |
| y | 17.1% |
| clarity | 12.8% |
| x | 9.5% |
| z | 8.2% |
| table | 5.4% |
| depth | 2.4% |
| color | 1.1% |
| cut | 0.3% |

**Key Insight**: Carat weight (43.2%) is by far the strongest predictor, followed by physical dimensions and clarity.

---

## 📊 Model Evaluation Metrics

### R² Score (Coefficient of Determination)
- Measures how well the model explains price variation
- Range: 0 to 1 (higher is better)
- Random Forest: 0.9816 means it explains 98.16% of price variation

### RMSE (Root Mean Squared Error)
- Average prediction error in dollars
- Random Forest: $540.63 average error
- Relative to price range ($326-$18,823), this is very accurate

---

## 📁 Visualizations Generated

1. **model_comparison.png** - Bar chart comparing R² scores of all 3 models
2. **feature_importance.png** - Horizontal bar chart showing feature importance scores
3. **rmse_comparison.png** - Bar chart comparing prediction errors of all 3 models
4. **prediction_actual.png** - Scatter plot showing predicted vs actual prices (with reference line)
5. **residuals.png** - Scatter plot showing prediction errors distribution

---

## 🎯 Key Findings

1. **Carat Weight Dominates**: Weight accounts for 43.2% of price variation
   - Heavier diamonds cost exponentially more
   - Most important factor for buyers

2. **Physical Dimensions Matter**: x, y, z dimensions (17.1% + 9.5% + 8.2%)
   - Correlated with carat weight
   - Larger stones cost more

3. **Clarity is Important**: 12.8% importance
   - Fewer imperfections = higher price
   - Noticeable impact on value

4. **Cut, Color Less Critical**: Combined only 1.4% importance
   - Still matter, but less than weight and clarity
   - Buyers prioritize weight over aesthetics

5. **Model Accuracy**: Random Forest achieves 98.16% R²
   - Predictions typically within $500-$600 of actual price
   - Reliable for practical use

---

## 📈 Model Performance Analysis

### Prediction Accuracy
- Points in scatter plot cluster tightly around diagonal reference line
- Indicates excellent prediction accuracy
- Only minor outliers (rare diamonds with unusual characteristics)

### Error Distribution
- Residuals randomly scattered around zero line
- No systematic bias (not consistently over/under predicting)
- Error magnitude increases with price (acceptable for regression)

---

## 🚀 How to Run

1. Install required libraries:
   ```bash
   pip install pandas scikit-learn numpy seaborn matplotlib
   ```

2. Run the Python script:
   ```bash
   python diamond_price_prediction.py
   ```

3. Charts will be saved as PNG files in the same directory:
   - model_comparison.png
   - feature_importance.png
   - rmse_comparison.png
   - prediction_actual.png
   - residuals.png

---

## 💡 What I Learned

- **Regression vs Classification**: Understanding the difference between predicting continuous values vs categories
- **Model Comparison**: How different algorithms (linear, tree-based, ensemble) perform differently
- **Feature Importance**: Quantifying which input factors drive predictions
- **Model Evaluation**: Using R² and RMSE to measure regression model performance
- **Visualization**: Creating meaningful charts to communicate findings
- **Data Encoding**: Converting categorical variables to numerical format

---

## 📚 Future Improvements

- **Hyperparameter Tuning**: Optimize Random Forest parameters (max_depth, min_samples_leaf)
- **Cross-Validation**: Use k-fold validation for more robust evaluation
- **Feature Engineering**: Create polynomial features or interaction terms
- **Outlier Detection**: Identify and handle unusual diamonds separately
- **Ensemble Methods**: Combine multiple models (Gradient Boosting, XGBoost)
- **API Deployment**: Create a web service to make predictions

---

## 📁 Project Structure

```
diamond-price-prediction/
├── README.md                    # This file
├── diamond_price_prediction.py  # Main Python code
├── model_comparison.png         # Model accuracy comparison
├── feature_importance.png       # Feature importance chart
├── rmse_comparison.png          # Model error comparison
├── prediction_actual.png        # Predicted vs actual prices
└── residuals.png                # Prediction errors distribution
```

---

## 👤 Author

Praj Maghiman V  
BTech - CSE (Data Science)  
Date: October 2025

---

## 🎓 Key Takeaways

This project demonstrates:
- Complete ML pipeline for regression problems
- Data preprocessing and categorical encoding
- Training and comparing multiple models
- Feature importance analysis
- Comprehensive model evaluation
- Professional data visualization

The Random Forest model successfully predicts diamond prices with 98.16% accuracy, proving that machine learning can effectively model complex pricing relationships in real-world datasets.

---

**Happy Learning!** 🚀
