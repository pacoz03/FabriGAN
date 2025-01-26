#Bringing in rest of the dependencies
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def load_dataset():
    ds, info = tfds.load('fashion_mnist', split='train', as_supervised=True, with_info=True)
    return ds, info

def display_samples(ds):
    # Converte il dataset in un iteratore di array NumPy
    dataiterator = tfds.as_numpy(ds)

    fig, ax = plt.subplots(ncols=16, figsize=(20, 20))
    # Itera quattro volte per ottenere le immagini
    for idx, (image, label) in enumerate(dataiterator):
        if idx == 16:
            break
        # Visualizza l'immagine
        ax[idx].imshow(np.squeeze(image), cmap='gray')
        ax[idx].title.set_text(label)
    plt.show()


def load_dataset_cnn():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # Ridimensiona le immagini per includere il canale dei colori
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Normalizza i valori dei pixel
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)


if __name__ == "__main__":
    dataset, info = load_dataset()
    display_samples(dataset)