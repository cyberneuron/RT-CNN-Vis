import tensorflow as tf

def getNetwork(name="ResNet50"):
    ph = tf.placeholder(tf.float32, shape=(None,224, 224,3),name="cnnInput")
    nn = getattr(tf.keras.applications,name)
    nn = nn(
        include_top=True,
        weights='imagenet',
        input_tensor=ph,
        input_shape=None,
        pooling=None,
        classes=1000
        )
    return nn,ph
