# FabriGAN

<div align="center">
  <img src="immagine.jpg" alt="Descrizione dell'immagine" width="310" height="300">
</div>

## Introduction

This project focuses on implementing a Generative Adversarial Network (GAN) to generate realistic clothing items using the Fashion MNIST dataset. The GAN aims to learn the underlying patterns of the dataset and create new, visually appealing examples of clothing items such as shirts, shoes, and dresses.

---

## Objectives

- Develop and train a GAN to generate high-quality clothing items.
- Utilize the Fashion MNIST dataset as a foundation for training the model.
- Leverage TensorFlow and other tools to streamline the development and visualization process.

---

## Installation

To get started, follow the steps below:

### 1. Install Dependencies

Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
The requirements.txt file includes:
```
TensorFlow
Matplotlib
Other necessary libraries
2. Clone the Repository
Clone this repository to your local machine:

bash
Copia
Modifica
git clone https://github.com/your-repository-url.git
cd your-repository-name
3. Set Up Your Environment
Ensure you have the necessary tools installed:

IDE: PyCharm (recommended)
Training Environment: Google Colab for GPU support
4. Run the Training Script



Navigate to the dataset/ directory and execute the training script:

```bash
python model.py
```
Dataset
Fashion MNIST
The Fashion MNIST dataset is used as the foundation for training the GAN. It contains 70,000 grayscale images of 28x28 pixels, categorized into 10 classes:

- T-shirts/tops
- Trousers
- Pullovers
- Dresses
- Coats
- Sandals
- Shirts
- Sneakers
- Bags
- Ankle boots

The dataset is preloaded in TensorFlow, making it easy to integrate into the pipeline.

### Technologies and Tools
TensorFlow: For building and training the GAN.
Matplotlib: For visualizing the generated clothing items and model performance.
Google Colab: For training the GAN using GPU acceleration.
PyCharm: As the primary IDE for local development.

### Project Structure
The project is organized as follows:
```bash
├── dataset
│   ├── model.py        # Contains the GAN implementation and training script
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation
```
Ensure that all dependencies are installed as described in the Installation section.
Execute the model.py script to train the GAN and generate new clothing items. 

Outputs will be saved, it includes:
1. model checkpoints during training every 150 epochs
2. generated images every 25 epochs
3. loss plots for the generator and discriminator



License
This project is licensed under the MIT License. See the LICENSE.md file for details.