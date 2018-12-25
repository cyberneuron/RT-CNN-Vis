import argparse
import itertools
import os

import tensorflow as tf
import cv2
import numpy as np

from networks import getNetwork
from utils import getConvOutput

def gradCam(y,A, reduce_max=True):
    """
    y,A according to paper
    """
    if reduce_max:
        # print("max reduced")
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

def gradCamToHeatMap(cam,im):
    heatShape = im.shape[:2]
    heatmap = cv2.resize(cam[0],heatShape)
    colored = np.uint8(0.7*im+0.3*cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET))
    return heatmap, colored

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', default="sample_images/cat_dog.png",
                        help="Input image")
    parser.add_argument('-o','--output', default="sample_images/cat_dog_cam.png",
                        help="Output Image")
    parser.add_argument('-n','--network', default="ResNet50",
                        help="Network (VGG16,ResNet50 ...)")
    parser.add_argument('--convindex', default=-1,type=int,
                        help="Index of convolutional layer to use in the algorithm (-1 for last layer)")

    args = parser.parse_args()
    # args=parser.parse_args([])

    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    nn,ph = getNetwork(name=args.network,gradients="Standard")
    nn.summary()
    from  tensorflow.keras.applications.resnet50 import decode_predictions
    preSoftMax = nn.output.op.inputs[0]
    A = getConvOutput(nn,args.convindex)
    top_k = 2
    values,indecies = tf.math.top_k(preSoftMax,k=top_k)
    # gradCamT = gradCam(preSoftMax[...,indecies[k]],A)
    # gradCamT_top_list = [gradCam(preSoftMax[...,indecies[k]],A,reduce_max=False) for k in range(top_k)]
    gradCamT_top_list = [gradCam(values[...,k],A,reduce_max=False) for k in range(top_k)]

    im = cv2.imread(args.input)
    im = cv2.resize(im,(224,224))
    # res = sess.run(gradCamT, {ph: [im]})
    res = sess.run(gradCamT_top_list, {ph: [im]})
    filename, file_extension = os.path.splitext(args.output)
    for k in range(top_k):
        heatmap,colored = gradCamToHeatMap(res[k], im)
        cv2.imwrite(f"{filename}_{k}{file_extension}", colored)
    # with StreamReader("http://192.168.16.101:8081/video") as cap:
    #
    #     for frame,num in zip(cap.read(),itertools.count()):
    #         im = cv2.resize(frame,(224,224))
    #         res = sess.run(gradCamT,{ph:[im]})
    #         heatmap,colored = gradCamToHeatMap(res,im)
    #         cv2.imshow("cam",colored)
    #         cv2.imwrite(f"cams/{num}.jpeg",colored)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
