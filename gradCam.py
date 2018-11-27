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
    gradCamRes = tf.reduce_mean(weightedFeatureMap,axis=[-1])
    # x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
    # gradCamRes = gradCamRes/ tf.sqrt(tf.mean(tf.square(gradCamRes))+1e-5)
    normalized = tf.nn.l2_normalize(gradCamRes)
    return normalized

if __name__ == "__main__":
    import cv2
    from networks import getNetwork
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    nn,ph = getNetwork(name="ResNet50")
    nn.summary()
    preSoftMax = nn.output.op.inputs[0]
    lastConv = nn.layers[-6 ]
    A = lastConv.output
    preSoftMax,A
    gradCamT = gradCam(preSoftMax,A)
    normalized = tf.nn.l2_normalize(gradCamT)
    normalized
