# Temperature Prediction Using MLP

This project demonstrates how to use a Multi-Layer Perceptron (MLP) for temperature prediction using time-series data. It utilizes TensorFlow/Keras for building and training the model, as well as simulated data for demonstration purposes.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Results](#results)
- [Customization](#customization)
- [License](#license)
- [Repository](#repository)

---

## Introduction

The goal of this project is to predict temperature values based on past observations using a sliding window technique. The model is a Multi-Layer Perceptron (MLP), which is trained on normalized time-series data. The code demonstrates preprocessing, model training, and evaluation.

---

## Features

- Simulated or real-world temperature data support.
- Flexible preprocessing with sliding windows.
- MLP with customizable architecture.
- Visualization of actual vs. predicted temperature trends.

---

## How It Works

1. **Simulate or Input Data**:
   Generate sinusoidal temperature data or replace it with real-world temperature data.

2. **Preprocess the Data**:
   Use a sliding window to create input-output pairs and normalize the data.

3. **Define and Train the MLP Model**:
   Train a model with two hidden layers to predict future temperatures.

4. **Evaluate and Visualize**:
   Measure model performance using Mean Absolute Error (MAE) and plot actual vs. predicted values.

---

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- scikit-learn
- Matplotlib

---

## Installation

To install the required libraries, run the following command:

```bash
pip install numpy tensorflow scikit-learn matplotlib
```

To clone this repository, run:

```bash
git clone https://github.com/Zero-Asif/IML.git
cd IML/MLP
```

To execute the script:

```bash
python temperature_prediction.py
```

---

## Results

The model evaluates the Mean Absolute Error (MAE) on the test set and generates a plot comparing actual vs. predicted temperature values.

---

## Customization

- **Window Size**: Modify `window_size` for different prediction horizons.
- **Model Architecture**: Add or remove layers, or adjust neuron counts as needed.
- **Data Input**: Replace simulated data with real-world temperature data.

---

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

## Repository

The complete code and updates for this project can be found in the following GitHub repository:

[Temperature Prediction Using MLP](https://github.com/Zero-Asif/IML/blob/main/MLP/temperature_prediction.py)

---

## Author

- **Asifuzzaman**  
- GitHub: [Zero-Asif](https://github.com/Zero-Asif)

---


