#Bringing in rest of the dependencies
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import numpy as np

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



if __name__ == "__main__":
    dataset, info = load_dataset()
    display_samples(dataset)