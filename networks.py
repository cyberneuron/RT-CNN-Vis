import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

tf.keras.activations.relu = tf.nn.relu

@tf.RegisterGradient("Customlrn")
def _CustomlrnGrad(op, grad):
    return grad

# register Relu gradients
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))



def getNetwork(name="ResNet50",gradients="relu"):
    graph = tf.get_default_graph()
    gradients_overwrite_map = {'Relu': 'GuidedRelu', 'LRN': 'Customlrn'} if gradients else {}
    with graph.gradient_override_map(gradients_overwrite_map):
        knownNets = ['DenseNet121',
             'DenseNet169',
             'DenseNet201',
             'InceptionResNetV2',
             'InceptionV3',
             'MobileNet',
             'MobileNetV2',
             'NASNetLarge',
             'NASNetMobile',
             'ResNet50',
              'VGG16',
             'VGG19',
             'Xception']
        # knownNets = ["ResNet50","VGG16","VGG19"]
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
