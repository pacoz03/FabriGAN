import tensorflow as tf
from PIL import Image

def scale_images(data):
    """
    Funzione di preprocessing per scalare le immagini da [0,255] a [0,1].
    """
    image = data['image']
    image = tf.cast(image, tf.float32) / 255.0
    return image


def crea_gif(lista_immagini, nome_output="animazione.gif", durata=6, loop=0):
    """
    Crea una GIF a partire da una lista di immagini.

    :param lista_immagini: Lista dei percorsi delle immagini.
    :param nome_output: Nome del file GIF generato.
    :param durata: Durata di ciascun frame in millisecondi.
    :param loop: Numero di loop (0 per loop infinito).
    """
    # Apri tutte le immagini e convertili nel formato adatto
    immagini = [Image.open(img) for img in lista_immagini]

    # Salva la prima immagine e aggiungi tutte le altre nel frame successivo
    immagini[0].save(
        nome_output,
        save_all=True,
        append_images=immagini[1:],
        duration=durata,
        loop=loop
    )
    print(f"GIF salvata come {nome_output}")