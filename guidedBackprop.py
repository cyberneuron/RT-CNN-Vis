# deconv gradients
from tensorflow.python.ops import gen_nn_ops
import tensorflow as tf
from collections import namedtuple, OrderedDict


BackPropTensors = namedtuple("BackpropTensors",["output","selection"])

def guidedBackprop(neuronOfInterest,nnInput):
    vis = tf.gradients(neuronOfInterest,nnInput)
    return vis

def registerConvBackprops(convOuts,nnInput,normalize=True,reduceMax=True):
    backprops = OrderedDict()
    for name,T in convOuts.items():
        x = tf.Variable(0)

        mapOfInterest = T[...,x]
        if reduceMax:
            mapOfInterest = tf.reduce_max(mapOfInterest)
        # constuct grad according to it
        print(f"Registering convolution layer backprop vis for  {name}:{T}")
        gradT = guidedBackprop([mapOfInterest],nnInput)[0]
        if normalize:
            gradT = tf.nn.relu(gradT)
            gradT = gradT/(tf.reduce_max(gradT)+1e-10)
        backprops[name] = gradT,x
    return backprops

def register_fc_backprops(fc_outs,nn_input,normalize=True):
    backprops = OrderedDict()
    for name,T in fc_outs.items():
        x = tf.Variable(0)
        neuron_of_interest = T[...,x]
        print(f"Registering fully connected layer backprop vis for {name}:{T}")
        gradT = guidedBackprop([neuron_of_interest],nn_input)[0]
        if normalize:
            gradT = tf.nn.relu(gradT)
            gradT = gradT/(tf.reduce_max(gradT)+1e-10)
        backprops[name] = BackPropTensors(gradT,x)
    return backprops

if __name__ == "__main__":
    from networks import getNetwork
    from streamReader import StreamReader
    import itertools
    import cv2
    import numpy as np
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    nn,ph = getNetwork(name="VGG16")
    nn.summary()
    preSoftMax = nn.output.op.inputs[0]
    neuronOfInterest = tf.reduce_max(preSoftMax,axis=-1,keepdims=True)
    neuronOfInterest
    guidedT = guidedBackprop(neuronOfInterest,ph)
    guidedT
    im = cv2.imread('sample_images/ManCoffee.jpeg')
    im = cv2.imread('sample_images/cat_dog.png')
    im = cv2.resize(im,(224,224))
    res = sess.run(guidedT,{ph:[im]})
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
