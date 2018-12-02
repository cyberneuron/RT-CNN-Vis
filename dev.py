import tensorflow as tf
import cv2
import numpy as np
import os
import argparse
from streamReader import StreamReader
import asyncio
import time
import math
import matplotlib.pyplot as plt

from gradCam import gradCam, gradCamToHeatMap
from guidedBackprop import registerConvBackprops
from networks import getNetwork
from maps import mapsToGrid


def getLayers(type='Conv2D'):
    return [i for i in graph.get_operations() if i.type.lower() == type.lower()]
    # return [i.type for i in graph.get_operations()]

def getLayersOutputs(type='Conv2D'):
    return [(i.outputs[0],i.name) for i in graph.get_operations() if i.type.lower() == type.lower()]
    # return [i.type for i in graph.get_operations()]

def getDenseTensors():
    layers = nn.layers
    return [layer.output for layer in layers if type(layer) is tf.keras.layers.Dense]

graph = tf.get_default_graph()
sess = tf.Session()
sess.as_default()

tf.keras.backend.set_session(sess)

nn, ph = getNetwork("VGG16")


convOutputs = getLayersOutputs()
convLayers = getLayers()

convGrids = {name: mapsToGrid(output[0]) for (output,name) in convOutputs}

convBackprops = registerConvBackprops(convOutputs,nn.input)

sess.run(tf.variables_initializer([convBackprops[name][1] for name in convBackprops ]))

gradCamA = nn.layers[-6].output
softmaxin = nn.output.op.inputs[0]
camT = gradCam(softmaxin,gradCamA)

print(getDenseTensors())

# TODO: fix this, (bad fast way to exit from programm)
close_main_loop = [False]

def rescale_img(image):
    img = np.uint8((1. - (image - np.min(image)) * 1. / (np.max(image) - np.min(image))) * 255)
    return img

async def main(ui=None, options={}):
    assert ui
    ui.fillLayers(convGrids.keys(), None)

    with StreamReader(args.stream) as cap:

        for frame in cap.read():
            currentGridName = ui.currentMap
            ui.loadRealImage(frame)
            raw_idx = ui.fmap.raw_idx

            frame = cv2.resize(frame,(224,224))
            frameToShow = frame.copy()
            frame = np.array([frame])
            gridTensor,(columns,rows), mapStack= convGrids[currentGridName]
            neuronBackpropT,neuronSelectionT = convBackprops[currentGridName]
            if raw_idx < len(mapStack):
                sess.run(neuronSelectionT.assign(raw_idx))
            aGrid, certainMap, cam, neuronBackprop = sess.run([gridTensor,mapStack[raw_idx],camT,neuronBackpropT],feed_dict={ph:frame})
            heatmap, coloredMap = gradCamToHeatMap(cam,frameToShow)
            cv2.imshow("gradCam",coloredMap)
            cv2.imshow("neuron-backprop",neuronBackprop[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ui.loadCell(rescale_img(certainMap))
            ui.loadMap(rescale_img(aGrid), (rows,columns))

            QApplication.processEvents()

            if close_main_loop[0]:
                break

    sys.exit(0)


import sys
import signal
from ui import Ui
from PyQt5.QtWidgets import QApplication
# from PyQt5.QtCore import QThread

def sigint_handler(*args):
    """Handler for the SIGINT signal."""
    # sys.stderr.write('\r')
    # if QMessageBox.question(None, '', "Are you sure you want to quit?",
    #                         QMessageBox.Yes | QMessageBox.No,
    #                         QMessageBox.No) == QMessageBox.Yes:
    close_main_loop[0] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="http://192.168.16.101:8081/video",
                        help="Input Video")
    parser.add_argument('--show', default=True,
                        help="Show output window")

    args = parser.parse_args()
    loop  = asyncio.get_event_loop()

    signal.signal(signal.SIGINT, sigint_handler)

    app = QApplication(sys.argv)
    ui = Ui()
    ui.show()

    loop.run_until_complete(main(ui=ui))


exit()
# writer = tf.summary.FileWriter("outputgraph", sess.graph)
# writer.close()
# unknownVisNodes = {}
# convGrids
# {'block1_conv1/Conv2D': <tf.Tensor 'concat_9:0' shape=(1792, 1792) dtype=float32>,
#  'block1_conv2/Conv2D': <tf.Tensor 'concat_19:0' shape=(1792, 1792) dtype=float32>,
#  'block2_conv1/Conv2D': <tf.Tensor 'concat_33:0' shape=(1232, 1344) dtype=float32>,
#  'block2_conv2/Conv2D': <tf.Tensor 'concat_47:0' shape=(1232, 1344) dtype=float32>,
#  'block3_conv1/Conv2D': <tf.Tensor 'concat_65:0' shape=(896, 896) dtype=float32>,
#  'block3_conv2/Conv2D': <tf.Tensor 'concat_83:0' shape=(896, 896) dtype=float32>,
#  'block3_conv3/Conv2D': <tf.Tensor 'concat_101:0' shape=(896, 896) dtype=float32>,
#  'block4_conv1/Conv2D': <tf.Tensor 'concat_126:0' shape=(644, 644) dtype=float32>,
#  'block4_conv2/Conv2D': <tf.Tensor 'concat_151:0' shape=(644, 644) dtype=float32>,
#  'block4_conv3/Conv2D': <tf.Tensor 'concat_176:0' shape=(644, 644) dtype=float32>,
#  'block5_conv1/Conv2D': <tf.Tensor 'concat_201:0' shape=(322, 322) dtype=float32>,
#  'block5_conv2/Conv2D': <tf.Tensor 'concat_226:0' shape=(322, 322) dtype=float32>,
#  'block5_conv3/Conv2D': <tf.Tensor 'concat_251:0' shape=(322, 322) dtype=float32>}



def illustrateDetections(frame,vggOut):
    preds = tf.keras.applications.vgg16.decode_predictions(vggOut)[0][0][1]
    cv2.putText(frameToShow,"{} {}".format(np.argmax(vggOut,axis=1),preds),(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1,cv2.LINE_AA)
