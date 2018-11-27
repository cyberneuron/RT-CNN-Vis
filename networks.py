import tensorflow as tf

def getNetwork(name="ResNet50"):
    knownNets = ["ResNet50","VGG16","VGG19"]
    assert name in knownNets , "Network should be one of {}".format(knownNets)
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
