import numpy as np
from keras.models import load_model
import os
import matplotlib.pyplot as plt

# Load datasets
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Check if the model exists
if os.path.exists('aapl_lstm_model.keras'):
    model = load_model('aapl_lstm_model.keras')
    print("Model loaded successfully.")
else:
    print("Model not found. Training a new one.")
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    
    # Define and train a new model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save('aapl_lstm_model.keras')  # Save the new model

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Generate predictions
y_pred = model.predict(X_test)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred.squeeze()) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Visualization: Actual vs. Predicted Values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Values', color='blue', alpha=0.7)
plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.7)
plt.title(f'Actual vs. Predicted Values\nMAPE: {mape:.2f}%')
plt.xlabel('Time Steps')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()

# Save predictions to a file for further analysis
np.save('y_pred.npy', y_pred)
