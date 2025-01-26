import tensorflow as tf
import os
import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.callbacks.callback import Callback
from dataset import utils
import glob

class ModelMonitor(Callback):
    def __init__(self,
                 num_img=9,
                 latent_dim=128,
                 output_dir='outputs'):
        """
        Args:
            num_img (int): Numero di immagini generate a fine epoca
            latent_dim (int): Dimensione del vettore di rumore
            output_dir (str): Directory radice dove salvare immagini, modelli e log
        """
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim

        # Inizializzo i percorsi locali
        self.output_dir = output_dir
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.losses_path = os.path.join(self.output_dir, 'losses.csv')

        # Creo le cartelle se non esistono
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append({
            'epoch': epoch + 1,
            'generator_loss': logs.get('g_loss', 0),
            'discriminator_loss': logs.get('d_loss', 0)
        })

        # Salva i loss su CSV ogni 5 epoche e genera immagini
        if (epoch + 1) % 5 == 0:
            pd.DataFrame(self.losses).to_csv(self.losses_path, index=False)
            self.save_images(epoch + 1)

        # Salva il modello ogni 5 epoche
        if (epoch + 1) % 5 == 0:
            self.save_model(epoch + 1)

    def on_train_end(self, logs=None):
        """
        Metodo chiamato a fine training.
        Salva il modello finale e il plot dei loss.
        """
        final_model_path = os.path.join(self.models_dir, 'model_final.keras')
        self.model.save(final_model_path)
        print(f"Modello finale salvato in: {final_model_path}")

        # Salva il CSV finale
        pd.DataFrame(self.losses).to_csv(self.losses_path, index=False)

        # Salva il grafico delle perdite
        self.save_loss_plot()

        #Crea gif di tutte le immagini generate
        utils.crea_gif(glob.glob(os.path.join(self.images_dir, 'generated_at_epoch_*.png')))



    def save_model(self, epoch):
        model_path = os.path.join(self.models_dir, f'model_at_epoch_{epoch}.keras')
        self.model.save(model_path)
        print(f"Modello salvato alla fine dell'epoca {epoch} in: {model_path}")

    def save_images(self, epoch):
        random_latent_vectors = tf.random.normal((self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors, training=False)

        # Le immagini sono in [0,1], le porto in [0,255]
        generated_images = generated_images.numpy() * 255.0
        generated_images = generated_images.astype('uint8')

        grid_size = int(self.num_img ** 0.5)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

        for i, ax in enumerate(axes.flat):
            if i < self.num_img:
                if generated_images.shape[-1] == 1:
                    #visualizza in scala di grigi
                    ax.imshow(generated_images[i, :, :, 0], cmap='gray')
                else:
                    ax.imshow(generated_images[i])
                ax.axis('off')

        image_path = os.path.join(self.images_dir, f'generated_at_epoch_{epoch}.png')
        plt.savefig(image_path)
        plt.close(fig)

        print(f"Immagini generate salvate in: {image_path}")

    def save_loss_plot(self):
        if self.losses:
            df_losses = pd.DataFrame(self.losses)
            plt.figure(figsize=(10, 6))
            plt.suptitle("Loss")
            plt.plot(df_losses['epoch'], df_losses['discriminator_loss'], label='d_loss')
            plt.plot(df_losses['epoch'], df_losses['generator_loss'], label='g_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plot_path = os.path.join(self.output_dir, 'images', 'loss_plot.png')

            plt.savefig(plot_path)
            plt.close()

            print(f"Grafico delle perdite salvato in: {plot_path}")


def cnn_callback():
    return keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )