# Artificial neural network (ANN)

- You can read about ANN in details on provided links:
  - [Deep Learning Basics](https://github.com/xscotophilic/Machine-Learning-Basic-Lessons/tree/main/7%20Deep%20Learning)
  - [Artificial Neural Networks](https://github.com/xscotophilic/machine-learning-lessons/blob/main/7%20Deep%20Learning/Artificial%20Neural%20Networks%20(ANN)/README.md)

---

- Outline
  1. ANN Classification (artificial_neural_network_classification.ipynb) (For Prediction of single image - upload test images on colab.)
  2. ANN Regression (artificial_neural_network_regression.ipynb)

## ANN Classification vs. ANN Regression

Both classification and regression problems can be solved using **Artificial Neural Networks (ANNs)**, but they have different objectives and structures.

| Feature               | **ANN for Classification** | **ANN for Regression** |
|----------------------|--------------------------|----------------------|
| **Objective**        | Assigns inputs to predefined categories (e.g., cat vs. dog). | Predicts a continuous numerical value (e.g., house price). |
| **Output Type**      | Discrete labels (e.g., 0 or 1, or multiple classes). | Continuous values (e.g., 23.7, 100.5). |
| **Activation Function in Output Layer** | **Sigmoid** (binary), **Softmax** (multiclass). | **Linear** (or no activation). |
| **Loss Function**    | Cross-Entropy Loss (for probability-based outputs). | Mean Squared Error (MSE) or Mean Absolute Error (MAE). |
| **Example Use Cases** | Image classification, spam detection, sentiment analysis. | Stock price prediction, weather forecasting, house price estimation. |
| **Decision Process** | Compares class probabilities and selects the highest. | Outputs a single numeric value as the prediction. |
