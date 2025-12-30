# Wine Quality Prediction using Artificial Neural Networks

This project implements various Artificial Neural Network (ANN) architectures to predict wine quality based on physiochemical properties. It utilizes regression models to analyze both red and white wine datasets, aiming to find the most efficient network structure for accurate quality estimation.

### Project Overview
The core objective of this study is to apply deep learning techniques to the "Wine Quality" dataset. The project explores the relationship between 11 independent variables (such as acidity, residual sugar, pH, and alcohol content) and the dependent variable: wine quality (scored on a numeric scale).

### Key Features
*   Regression Modeling: Implements ANN regression where the output layer consists of a single neuron providing a continuous numeric value for quality.
*   K-Fold Cross-Validation: Uses 10-fold cross-validation to ensure model robustness and to provide a comprehensive evaluation of the training and testing sets.
*   Automated Preprocessing: Includes data shuffling, feature selection, and normalization to improve neural network convergence.
*   Optimized Training: Incorporates "Early Stopping" callbacks to prevent overfitting by monitoring validation loss and stopping training at the optimal point.
*   Performance Evaluation: Models are evaluated using Root Mean Squared Error (RMSE) and Loss functions across multiple architectures.
*   Visualization: Includes integrated regression charts to compare predicted values against expected values (ground truth).

### Technologies and Libraries
*   Programming Language: Python
*   Deep Learning Framework: Keras / TensorFlow
*   Data Manipulation: Pandas, NumPy
*   Machine Learning Tools: Scikit-learn (KFold, Metrics)
*   Visualization: Matplotlib
*   Statistical Analysis: SciPy (Z-score)

### Neural Network Architecture
The project tests multiple architectures (ranging from 2 to 3 hidden layers). A typical configuration used in the implementation includes:
*   Input Layer: 11 neurons (corresponding to wine parameters).
*   Hidden Layers: Multiple Dense layers with ReLU activation (e.g., 30-35-30 neurons).
*   Output Layer: 1 neuron with linear activation for regression.
*   Optimizer: Adam
*   Loss Function: Mean Squared Error (MSE)

### Experimental Results
The study compared several architectures (RNA1r to RNA10a), analyzing the impact of layer depth, neuron count, and the number of epochs (up to 1000) on the final RMSE score. Detailed comparison tables for both red and white wine datasets are included in the documentation.

### Further Details
For a detailed analysis of the variables, statistical parameters, and a comparison of all tested neural architectures, please refer to the [documentation](https://github.com/emanuelco07/Wine_Quality_Prediction_using_Artificial_Neural_Networks/blob/main/Calitate_vinuri.pdf) in Romanian available in this repository.
