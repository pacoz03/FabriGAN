
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import dataset.dataset as ds
from dataset.models import build_model
from monitoring.callback import cnn_callback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def train_cnn():
    # Carica il dataset
    (train_images, train_labels), (test_images, test_labels) = ds.load_dataset_cnn()


    #Data Augmentation1
    """datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )"""

    #Data Augmentation2
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )


    # Adatta il generatore ai dati di addestramento
    datagen.fit(train_images)

    # Costruisci il modello
    model=build_model()

    # Callback per il monitoraggio
    callback= cnn_callback()

    # Addestra il modello
    batch_size = 64
    epochs = 50

    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        epochs=epochs,
        validation_data=(test_images, test_labels),
        callbacks=[callback]
    )

    return history,model



def stats(history):
    # Grafico dell'accuratezza
    plt.plot(history.history['accuracy'], label='Accuratezza Training')
    plt.plot(history.history['val_accuracy'], label='Accuratezza Validazione')
    plt.title('Andamento dell\'accuratezza durante l\'addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Accuratezza')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

    # Grafico della perdita
    plt.plot(history.history['loss'], label='Perdita Training')
    plt.plot(history.history['val_loss'], label='Perdita Validazione')
    plt.title('Andamento della perdita durante l\'addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Perdita')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


#history,model=train_cnn()

def test_cnn():
    # Carica il dataset
    (train_images, train_labels), (test_images, test_labels) = ds.load_dataset_cnn()

    # Costruisci il modello
    model = build_model()

    # Carica il modello addestrato
    model.load_weights('best_model.keras')

    # Prevedi le etichette per il set di test
    y_pred = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Stampa il rapporto di classificazione
    target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(classification_report(test_labels, y_pred_classes, target_names=target_names))

    # Prevedi le probabilità per ciascuna classe
    y_pred_proba = model.predict(test_images)

    # Ottieni le classi con la probabilità più alta
    y_pred = np.argmax(y_pred_proba, axis=1)
    # Crea la matrice di confusione
    cm = confusion_matrix(test_labels, y_pred)
    # Definisci i nomi delle classi del dataset Fashion MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Crea la visualizzazione della matrice di confusione
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Traccia la matrice di confusione
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.xticks(rotation=45)
    plt.title('Matrice di Confusione del Modello Fashion MNIST')
    plt.show()


if __name__ == "__main__":
    test_cnn()