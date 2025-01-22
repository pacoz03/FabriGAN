
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.losses import BinaryCrossentropy
from models import build_discriminator, build_generator
#from monitoring.callback import ModelMonitor
from utils import scale_images,crea_gif
import glob
import os

class FashionGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        # 1) Ottengo i batch di immagini reali
        real_images = batch

        # 2) Genero immagini fake con il generatore
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # ================== TRAIN DISCRIMINATOR ================== #
        with tf.GradientTape() as d_tape:
            # Classificazione delle immagini reali e fake
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)

            # Unisco i risultati di yhat_real e yhat_fake per calcolare la loss
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Creo le label: 0 per reale, 1 per fake
            y_realfake = tf.concat([
                tf.zeros_like(yhat_real),
                tf.ones_like(yhat_fake)
            ], axis=0)

            # Aggiungo un po' di rumore alle uscite per stabilizzare il training
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake_noisy = y_realfake + tf.concat([noise_real, noise_fake], axis=0)

            # Calcolo la loss del discriminatore
            total_d_loss = self.d_loss(y_realfake_noisy, yhat_realfake)

        # Backprop sul discriminatore
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # ================== TRAIN GENERATOR ================== #
        with tf.GradientTape() as g_tape:
            # Genero nuove immagini
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            # Valuto il discriminatore su queste immagini
            predicted_labels = self.discriminator(gen_images, training=False)

            # Loss del generatore: vogliamo "ingannare" il discriminatore
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        # Backprop sul generatore
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {
            "d_loss": total_d_loss,
            "g_loss": total_g_loss
        }



def main():
    # Caricamento dataset
    ds = tfds.load('fashion_mnist', split='train')
    ds = ds.map(scale_images)
    ds = ds.cache()
    ds = ds.shuffle(60000)
    ds = ds.batch(128)
    ds = ds.prefetch(64)

    # Costruzione generator e discriminator
    generator = build_generator()
    discriminator = build_discriminator()

    # Creazione istanza FashionGAN
    fashgan = FashionGAN(generator=generator, discriminator=discriminator)

    # Ottimizzatori e loss
    g_opt = Adam(learning_rate=0.0001)
    d_opt = Adam(learning_rate=0.00001)
    g_loss = BinaryCrossentropy()
    d_loss = BinaryCrossentropy()

    # Compilazione del modello
    fashgan.compile(
        g_opt=g_opt,
        d_opt=d_opt,
        g_loss=g_loss,
        d_loss=d_loss
    )
    # Training
    fashgan.fit(ds, epochs=1)
"""
    #visualizzazione di immagini generate
    generated_images = fashgan.generator(tf.random.normal((9, 128, 1)), training=False)

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(generated_images[i*3+j])
            ax[i, j].axis('off')

    plt.show()"""



if __name__ == "__main__":
    #main()

    crea_gif(glob.glob(os.path.join("..\\outputs\\training2\\images", 'generated_at_epoch_*.png')),durata=10)
