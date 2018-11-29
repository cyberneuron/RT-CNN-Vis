# deconv gradients
from tensorflow.python.ops import gen_nn_ops
import tensorflow as tf



def guidedBackprop(neuronOfInterest,nnInput):
    vis = tf.gradients(neuronOfInterest,nnInput)
    return vis

def registerConvBackprops(convOuts,nnInput):
    backprops = {}
    for T, name in convOuts:
        x = tf.Variable(0)

        mapOfInterest = T[...,x]
        # constuct grad according to it
        gradT = guidedBackprop(mapOfInterest,nnInput)
        backprops[name] = gradT,x
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
