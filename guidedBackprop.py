# deconv gradients
from tensorflow.python.ops import gen_nn_ops
import tensorflow as tf

tf.keras.activations.relu
tf.nn.relu

tf.keras.activations.relu = tf.nn.relu


@tf.RegisterGradient("Customlrn")
def _CustomlrnGrad(op, grad):
    return grad

# register Relu gradients
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

def guidedBackprop(graph,neuronOfInterest,nnInput):
    ret = {}
    with graph.gradient_override_map({'Relu': 'GuidedRelu', 'LRN': 'Customlrn'}):
        vis = tf.gradients(neuronOfInterest,nnInput)
    return vis

def createGrad(graph,map,input):
    print("entering with")
    with graph.gradient_override_map({'Relu': 'GuidedRelu', 'LRN': 'Customlrn'}):
        print("creating grad node")
        grad = tf.gradients(map, input)
    return grad

if __name__ == "__main__":
    from networks import getNetwork
    from streamReader import StreamReader
    import itertools
    import cv2
    import numpy as np
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    with graph.gradient_override_map({'Relu': 'GuidedRelu', 'LRN': 'Customlrn'}):
        nn,ph = getNetwork(name="VGG16")
    nn.summary()
    preSoftMax = nn.output.op.inputs[0]
    neuronOfInterest = tf.reduce_max(preSoftMax,axis=-1,keepdims=True)
    neuronOfInterest
    graph = tf.get_default_graph()
    guidedT = guidedBackprop(graph,neuronOfInterest,ph)
    guidedT
    im = cv2.imread('sample_images/ManCoffee.jpeg')
    im = cv2.imread('sample_images/cat_dog.png')
    im = cv2.resize(im,(224,224))
    # for i in range(1000):

    # with graph.gradient_override_map({'Relu': 'GluidedRelu', 'LRN': 'Customlrn'}):
    res = sess.run(guidedT,{ph:[im]})
    res[0,570]
    res[0].shape
    res[0][0]
    cv2.imwrite("guided.jpg",res[0][0]*60000)
    with StreamReader("http://192.168.16.101:8081/video") as cap:

        for frame,num in zip(cap.read(),itertools.count()):
            im = cv2.resize(frame,(224,224))
            res = sess.run(gradCamT,{ph:[im]})
            heatmap,colored = gradCamToHeatMap(res,im)
            cv2.imshow("cam",colored)
            cv2.imwrite(f"cams/{num}.jpeg",colored)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
