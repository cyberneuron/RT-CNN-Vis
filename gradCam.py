import tensorflow as tf
import numpy as np
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
    normalized = tf.nn.l2_normalize(relu)
    # normalized = relu/tf.sqrt(tf.reduce_mean(tf.square(relu)+1e-5))
    # tf.square
    return normalized

if __name__ == "__main__":
    import cv2
    from networks import getNetwork
    import matplotlib.pyplot as plt
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    nn,ph = getNetwork(name="ResNet50")
    nn.summary()
    preSoftMax = nn.output.op.inputs[0]
    lastConv = nn.layers[-6 ]
    A = lastConv.output
    preSoftMax,A
    gradCamT = gradCam(preSoftMax,A)
    im = cv2.imread('sample_images/ManCoffee.jpeg')
    im = cv2.imread('sample_images/cat_dog.png')

    im = cv2.resize(im,(224,224))
    res = sess.run(gradCamT,{ph:[im]})
    heatmap = cv2.resize(res[0],(224,224))
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cv2.imwrite("gradcam.jpg",cam)
