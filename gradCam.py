import tensorflow as tf

def gradCam(y,A):
    """
    y,A according to paper
    """
    y = tf.reduce_max(y,axis=-1,keepdims=True)
    mapImportanceGradients = tf.gradients(y,A)
    importanceWeights = tf.reduce_mean(mapImportanceGradients, axis=[2,3],keepdims=True)
    # shape extended with one in beginning?
    weightsReshaped = importanceWeights[0]
    weightedFeatureMap = weightsReshaped*A
    reduced = tf.reduce_mean(weightedFeatureMap,axis=[-1])
    relu = tf.nn.relu(reduced)
    # x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
    # gradCamRes = gradCamRes/ tf.sqrt(tf.mean(tf.square(gradCamRes))+1e-5)
    # normalized = tf.nn.l2_normalize(relu)
    # normaliized = relu/tf.sqrt(tf.reduce_mean(tf.square(relu)+1e-5))
    normalized = relu/(tf.reduce_max(relu)+1e-12)
    # tf.square
    return normalized

import cv2
import numpy as np
def gradCamToHeatMap(cam,im):
    heatShape = im.shape[:2]
    heatmap = cv2.resize(cam[0],heatShape)
    colored = np.uint8(0.7*im+0.3*cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET))
    return heatmap, colored

if __name__ == "__main__":

    from networks import getNetwork
    import matplotlib.pyplot as plt
    from streamReader import StreamReader
    import itertools
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    nn,ph = getNetwork(name="VGG16")
    nn.summary()
    preSoftMax = nn.output.op.inputs[0]
    lastConv = nn.layers[-6 ]
    A = lastConv.output

    gradCamT = gradCam(preSoftMax,A)
    im = cv2.imread('sample_images/ManCoffee.jpeg')
    im = cv2.imread('sample_images/cat_dog.png')
    im = cv2.resize(im,(224,224))
    # for i in range(1000):
    res = sess.run(gradCamT,{ph:[im]})
    heatmap,colored = gradCamToHeatMap(res,im)

    cv2.imwrite("gradcam.jpg",colored)
    with StreamReader("http://192.168.16.101:8081/video") as cap:

        for frame,num in zip(cap.read(),itertools.count()):
            im = cv2.resize(frame,(224,224))
            res = sess.run(gradCamT,{ph:[im]})
            heatmap,colored = gradCamToHeatMap(res,im)
            cv2.imshow("cam",colored)
            cv2.imwrite(f"cams/{num}.jpeg",colored)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
