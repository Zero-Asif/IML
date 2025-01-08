# LSTM Hyperparameter Tuning for Time Series Forecasting

This project demonstrates how to build and tune a Long Short-Term Memory (LSTM) neural network for time-series forecasting using Keras Tuner. It uses the AirPassengers dataset, which contains monthly totals of airline passengers.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Code Structure](#code-structure)
6. [How to Run](#how-to-run)
7. [Model Tuning and Training](#model-tuning-and-training)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Acknowledgments](#acknowledgments)

---

## Project Overview
This project builds a neural network model for time series prediction, focusing on tuning hyperparameters using Keras Tuner. It applies:
- **LSTM layers** for sequential modeling.
- **Dropout layers** to prevent overfitting.
- **Dense layers** to enhance prediction performance.
- **Hyperparameter optimization** to determine the best architecture.

---

## Dataset
The dataset used is the **AirPassengers.csv** file containing monthly airline passenger totals from 1949 to 1960.

Dataset Columns:
- **Month**: Date of observation (in YYYY-MM format).
- **#Passengers**: Number of airline passengers in that month.

Source: [AirPassengers Dataset](https://github.com/Zero-Asif/data/blob/main/AirPassengers.csv)

---

## Prerequisites
Ensure the following are installed:
- Python 3.x
- TensorFlow 2.x
- Pandas
- Scikit-learn
- Keras Tuner

---

## Installation
```bash
pip install tensorflow pandas scikit-learn keras-tuner
```

---

## Code Structure
- **load_data()**: Loads and preprocesses the dataset.
- **build_model()**: Defines the LSTM model architecture with hyperparameter tuning.
- **Hyperparameter Tuning**: Uses Keras Tuner's Hyperband algorithm.
- **Model Training and Evaluation**: Fits the best model and evaluates performance.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Zero-Asif/IML.git
   cd IML/LSTM
   ```
2. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the script:
   ```bash
   python lstm.py
   ```

---

## Model Tuning and Training
### Hyperparameter Tuning
The model architecture is tuned with the following parameters:
- **LSTM Layers**: Number of layers (1 to 3).
- **Units per Layer**: 32 to 256 units in steps of 32.
- **Activation Functions**: ReLU, ELU, or Sigmoid.
- **Dropout Rates**: Between 0.1 and 0.5.
- **Dense Layers**: 1 to 3 with variable units.
- **Learning Rates**: 0.001, 0.01, or 0.0001.

### Training
The final model is trained using:
- **Early Stopping**: Stops training if validation loss does not improve for 5 epochs.
- **Validation Split**: 20% of training data used for validation.
- **Epochs**: 50 (maximum).

---

## Evaluation
The model is evaluated on:
- **Loss**: Mean Squared Error (MSE) or Mean Absolute Error (MAE).
- **Metrics**: Mean Absolute Error (MAE).

---

## Results
- **Best Hyperparameters**: Displays the best parameters found during tuning.
- **Test Loss and MAE**: Final evaluation metrics on the test set.

---

## Author

- **Asifuzzaman**  
- GitHub: [Zero-Asif](https://github.com/Zero-Asif)

---

## Acknowledgments
- **Dataset Source**: Available publicly on GitHub.
- **Libraries**: TensorFlow, Pandas, and Keras Tuner.
- Special thanks to Keras Tuner developers for the hyperparameter optimization framework.

---




