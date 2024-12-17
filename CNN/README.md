# Electric Production Forecasting using a Convolutional Neural Network (1D-CNN)

This repository contains a project to predict **electric production** using a 1D Convolutional Neural Network (1D-CNN) built with TensorFlow/Keras. The dataset is sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and processed to train, validate, and test the model.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Code Execution](#code-execution)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

---

## Overview

This project implements a machine learning pipeline for time series prediction:
1. Loads and processes the dataset.
2. Splits data into training, validation, and test sets.
3. Reshapes input data for 1D CNN compatibility.
4. Builds a neural network model using **Conv1D** layers to capture temporal dependencies.
5. Trains the model, evaluates its performance, and visualizes the predicted values against the true values.

---

## Dataset

The dataset used is **Electric Production Data**, available in a CSV format.

- **Dataset Link**: [Electric Production CSV](https://raw.githubusercontent.com/Zero-Asif/data/refs/heads/main/electric-production.csv)
- **Columns**:
   - The first column contains time-based index data.
   - The subsequent columns represent production values over time.

---

## Model Architecture

The implemented Convolutional Neural Network includes the following layers:

1. **Conv1D Layer**: Extracts features with 64 filters and `ReLU` activation.
2. **Dropout Layer**: Prevents overfitting with a dropout rate of 0.2.
3. **Conv1D Layer**: Second convolutional layer with 32 filters.
4. **Flatten Layer**: Converts data into a 1D feature vector.
5. **Dense Layer**: Fully connected layer with 50 neurons and `ReLU` activation.
6. **Output Dense Layer**: Outputs the predicted value.

Optimizer: **Adam** (learning rate = 0.001)  
Loss Function: **Mean Squared Error (MSE)**  
Evaluation Metric: **Mean Absolute Error (MAE)**  

---

## Dependencies

To run this project, ensure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib tensorflow scikit-learn

---

## Code Execution

### 1. Clone the repository:

   ```bash
   git clone https://github.com/Zero-Asif/IML.git && cd IML/CNN && ls CNNmodel.py

### 2. Run the Python script:

   ```bash
   python CNNmodel.py

### 3. Outputs

- **Training and validation loss metrics**.  
- **Test loss and Mean Absolute Error (MAE)**.  
- **Visualization plots**:  
  - True values  
  - Predicted values  
  - Comparison of true and predicted values  

---

## Results

The model outputs:

- **Test Loss**: Final mean squared error on the test set.  
- **Test MAE**: Mean Absolute Error on the test set.  

Visualizations help analyze the performance and accuracy of predictions.

---

## Visualization

Three plots are generated:

1. **True Values**: Visualization of actual values from the test set.  
2. **Predicted Values**: Visualization of predicted values by the model.  
3. **Comparison**: Overlay of true and predicted values for comparison.  

---

## License

This project is licensed under the **MIT License**.

---

## Author

- **Asifuzzaman**  
- GitHub: [Zero-Asif](https://github.com/Zero-Asif)

---

## Acknowledgements

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)  
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
