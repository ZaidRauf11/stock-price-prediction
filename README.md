# 📈 Stock Price Prediction using Deep Learning

This project demonstrates how to build a deep learning model that predicts **stock closing prices** based on historical data (symbol, open, high, low, and volume).

The pipeline includes:

* Data preprocessing (One-Hot Encoding for stock symbols, scaling for numeric features)
* Model training with TensorFlow/Keras
* Evaluation of performance on test data
* Making predictions on new stock samples
* Visualization of **Actual vs Predicted stock prices**

---

## 🚀 Project Workflow

### 1. Import Libraries

We use:

* **pandas, numpy** → for data handling
* **scikit-learn** → preprocessing & train/test split
* **tensorflow/keras** → deep learning model
* **matplotlib** → visualization

---

### 2. Load Dataset

```python
df = pd.read_csv("Stock prices.csv")
df.head()
```

The dataset contains:

* `symbol` → Stock ticker (e.g., AAPL)
* `open`, `high`, `low`, `volume` → Features
* `close` → Target value (closing price)

---

### 3. Preprocessing

* One-Hot Encode the `symbol` column
* Standardize numerical columns (`open`, `high`, `low`, `volume`)
* Scale target `y` values

---

### 4. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

### 5. Model Architecture

A fully-connected feedforward neural network:

* Dense(512, ReLU)
* Dense(128, ReLU)
* Dense(64, ReLU)
* Dense(1, Linear) → Output predicted closing price

```python
model = models.Sequential()
model.add(layers.Dense(512, input_dim=X_transformed.shape[1], activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
```

---

### 6. Training

```python
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_transformed, y_transformed, epochs=50, verbose=1)
```

---

### 7. Evaluation

```python
test_loss, test_mae = model.evaluate(X_test_transformed, y_test_transformed)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test MAE: {test_mae}")
```

---

### 8. Predictions

* Predict closing prices for test set
* Inverse transform results back to **real stock prices**
* Compare first 10 predictions against actual values

---

### 9. New Sample Prediction

Example:

```python
test_sample = pd.DataFrame([{
    'symbol': 'AAPL',
    'open': 100,
    'high': 105,
    'low': 98,
    'volume': 5000000
}])

y_pred_real = y_scaler.inverse_transform(model.predict(transformer.transform(test_sample)))
print("Predicted Stock Price:", y_pred_real[0][0])
```

---

### 10. Visualization

Plot actual vs predicted stock prices for the first 100 samples:

```python
plt.plot(y_true[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.show()
```

---

## 📊 Results

* Model learns patterns between stock features and closing price.
* Visualization helps compare predictions with actual prices.
* Can predict stock prices for **new unseen input** (e.g., future market data).

---

<img width="1169" height="609" alt="image" src="https://github.com/user-attachments/assets/2e639a25-fed7-4738-bb83-029dea4f34ed" />

---

## 🛠 Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

---

## 📌 How to Run

1. Clone the repo
2. Place your `Stock prices.csv` file in the project directory
3. Run the notebook / script step by step
4. Check predictions & plots

