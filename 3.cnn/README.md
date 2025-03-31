# Convolutional neural network (CNN)

- The code file itself contains description of each block.
- You can read about CNN concepts in details on provided links:
  - [Deep Learning Basics](https://github.com/xscotophilic/Machine-Learning-Basic-Lessons/tree/main/7%20Deep%20Learning)
  - [Convolutional Neural Network](https://github.com/xscotophilic/machine-learning-lessons/blob/main/7%20Deep%20Learning/Convolutional%20Neural%20Networks%20(CNN)/README.md)
- Extra Read:
  - [Batch Normalization In Neural Networks Explained (Algorithm Breakdown) - on towardsdatascience](https://towardsdatascience.com/batch-normalization-explained-algorithm-breakdown-23d2794511c)
  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

---

- Outline
  1. Convolutional neural network (dataset: CIFAR10)(cnn_cifar10.ipynb) (For Prediction of single image - upload cifar_test images on colab.)
  2. Convolutional neural network (dataset: fashion_mnist)(cnn_fashion_mnist.ipynb) (For Prediction of single image - upload fashion_mnist_tests images on colab.)

## CNN for CIFAR-10 vs. CNN for Fashion-MNIST  

Both **CIFAR-10** and **Fashion-MNIST** datasets are commonly used for training **Convolutional Neural Networks (CNNs)**, but they differ in complexity and characteristics.  

| Feature               | **CNN for CIFAR-10** | **CNN for Fashion-MNIST** |
|----------------------|--------------------|------------------------|
| **Dataset Type**    | 60,000 color images of **10 objects** (airplane, car, bird, etc.). | 70,000 grayscale images of **10 fashion items** (shoes, shirts, bags, etc.). |
| **Image Size**      | **32 × 32 × 3** (RGB) | **28 × 28 × 1** (Grayscale) |
| **Color Channels**  | 3 (Red, Green, Blue) | 1 (Grayscale) |
| **Complexity**      | More complex due to color and high variation. | Simpler, with clear shapes and low texture variation. |
| **CNN Architecture Complexity** | Needs **deeper networks** with **more filters** to capture color and detailed object shapes. | Can work with **shallower networks** as grayscale images have fewer features to learn. |
| **Data Augmentation** | Strongly recommended (flipping, rotation, brightness adjustment) to improve generalization. | Light augmentation (flipping, cropping) can help but isn’t always necessary. |
| **Example CNN Model** | VGG-16, ResNet, EfficientNet | Simple CNN with a few convolutional layers (LeNet-style) |
| **Use Case** | Real-world object recognition (cars, animals, etc.). | Fashion product classification (e-commerce, clothing stores). |
