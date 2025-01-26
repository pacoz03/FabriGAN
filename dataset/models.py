from keras.src.models import Sequential
#Bringing in the layers for the neural network.
from keras.src.layers import Conv2D, Dense, Reshape, Flatten, LeakyReLU, Dropout, UpSampling2D, Input, MaxPooling2D, BatchNormalization

def build_generator():
    """
    Costruisce la rete generatrice.
    """
    model = Sequential()

    model.add(Input(shape=(28, 28, 1)))  # Definisci l'input shape all'inizio


    # Primo layer Dense: genera un volume 7x7x128 a partire da un input di dimensione 128
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))

    # Upsampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 1
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 2
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Layer di output: 1 canale, attivazione sigmoid
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))
    return model


def build_discriminator():
    """
    Costruisce la rete discriminante.
    """
    model = Sequential()

    model.add(Input(shape=(28, 28, 1)))  # Definisci l'input shape all'inizio


    # Primo blocco di convoluzione
    model.add(Conv2D(32, 5, input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Secondo blocco di convoluzione
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Terzo blocco di convoluzione
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Quarto blocco di convoluzione
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Flatten + Dense di output
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model



def build_model():

    model = Sequential(name="MCNN15_Model")
    model.add(Input(shape=(28, 28, 1)))

    # Aggiungi strati convoluzionali con pooling meno frequente
    for i in range(15):
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        if i % 3 == 0:  # Applica il pooling ogni 3 strati
            model.add(MaxPooling2D((2, 2), padding='same'))
            model.add(Dropout(0.25))

    # Appiattisci e aggiungi strati densi
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model