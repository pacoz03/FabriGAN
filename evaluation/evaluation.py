from scipy.linalg import sqrtm
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
import keras

from model.model import FashionGAN


def calculate_fid(real_images, generated_images):
    # Carica un modello pre-addestrato per estrarre le caratteristiche
    model = load_model('best_model.keras')

    # Estrarre le caratteristiche
    act1 = model.predict(real_images)
    act2 = model.predict(generated_images)

    # Calcola media e covarianza
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calcola la distanza di Frechet
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Carica il modello completo (FashionGAN)
loaded_gan = keras.models.load_model('../outputs/training2/models/model_final.keras', custom_objects={'FashionGAN': FashionGAN})
# Ora puoi accedere al generator all’interno di loaded_gan
generator = loaded_gan.generator

(_, _), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Ridimensiona le immagini per includere il canale dei colori
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Normalizza i valori dei pixel
test_images =  test_images / 255.0

generated_images = generator(tf.random.normal((64, 128)), training=False).numpy()
generated_images = (generated_images + 1.0) / 2.0  # Normalizza a [0,1]

fid_score = calculate_fid(test_images, generated_images)
print(f'FID Score: {fid_score}')