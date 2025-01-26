# data_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.datasets import fashion_mnist


def load_data():
    """
    Carica il dataset Fashion MNIST e restituisce i set di addestramento e test.
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def explore_data(x, y, dataset_type='Addestramento'):
    """
    Esplora e visualizza le caratteristiche del dataset.

    Args:
        x (numpy.ndarray): Array delle immagini.
        y (numpy.ndarray): Array delle etichette.
        dataset_type (str): Tipo di dataset (es. 'Addestramento', 'Test').
    """
    print(f"--- Analisi del Dataset {dataset_type} ---")
    print(f"Numero di campioni: {x.shape[0]}")
    print(f"Dimensioni delle immagini: {x.shape[1]}x{x.shape[2]} pixel")
    print(f"Numero di classi: {len(np.unique(y))}")

    # Distribuzione delle classi
    class_counts = np.bincount(y)
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=class_counts, palette='viridis')
    plt.title(f'Distribuzione delle Classi nel Dataset {dataset_type}')
    plt.xlabel('Classe')
    plt.ylabel('Numero di Campioni')
    plt.xticks(rotation=45)
    plt.savefig(f'distribution_{dataset_type.lower()}.png')
    plt.show()

    # Visualizzazione di esempi per ciascuna classe
    plt.figure(figsize=(12, 12))
    for i, class_name in enumerate(class_names):
        # Seleziona una immagine per classe
        idx = np.where(y == i)[0][0]
        plt.subplot(5, 2, i + 1)
        plt.imshow(x[idx], cmap='gray')
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'examples_{dataset_type.lower()}.png')
    plt.show()


def main():
    # Caricamento dei dati
    (x_train, y_train), (x_test, y_test) = load_data()

    # Esplorazione del set di addestramento
    explore_data(x_train, y_train, dataset_type='Addestramento')

    # Esplorazione del set di test
    explore_data(x_test, y_test, dataset_type='Test')


if __name__ == "__main__":
    main()
