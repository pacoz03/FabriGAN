import tensorflow as tf

def scale_images(data):
    """
    Funzione di preprocessing per scalare le immagini da [0,255] a [0,1].
    """
    image = data['image']
    image = tf.cast(image, tf.float32) / 255.0
    return image