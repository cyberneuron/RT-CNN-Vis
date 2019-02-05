import os
import argparse
import asyncio
import time
import math
from collections import namedtuple, OrderedDict
import itertools

import tensorflow as tf
import cv2
import numpy as np

from streamReader import StreamReader
from gradcam import gradCam, gradCamToHeatMap
from guidedBackprop import registerConvBackprops, register_fc_backprops
from networks import get_network
from maps import mapsToGrid
from utils import get_outputs_from_graph, get_outputs_from_model, getConvOutput
from timed import timeit,Timer,FPS

parser = argparse.ArgumentParser()
parser.add_argument('--stream', default="http://192.168.16.101:8081/video",
# parser.add_argument('--stream', default="http://191.167.15.101:8079",
                    help="Video stram URI, webcam number or path to a video based on which the network is visualized")
# parser.add_argument('--show', default=True,
#                     help="Show output window")
parser.add_argument('--network', default="VGG16",
                    help="Network to visualise: One of built in keras applications (VGG16,ResNet50 ...) or path to .h5 file")
args = parser.parse_args()



graph = tf.get_default_graph()
sess = tf.Session()
sess.as_default()
tf.keras.backend.set_session(sess)

nn, ph = get_network(args.network)

print(nn.summary())

# conv_outputs = get_outputs_from_graph(type='Conv2D')
conv_outputs = get_outputs_from_model(nn,layer_type='Conv2D')
assert conv_outputs, "Provided network has no Convolutional layers, hence I have no idea what to visualize"



conv_grids = OrderedDict( (name, mapsToGrid(output[0])) for name, output in conv_outputs.items())



convBackprops = registerConvBackprops(conv_outputs,nn.input)

fc_outputs = get_outputs_from_model(nn,layer_type="Dense")
fc_backprops = register_fc_backprops(fc_outputs,nn.input)



sess.run(tf.variables_initializer([convBackprops[name][1] for name in convBackprops ]))
sess.run(tf.variables_initializer([fc_backprops[name].selection for name in fc_backprops ]))

if fc_outputs:
    # GradCam is possible if there are fully connected layers
    gradCamA = getConvOutput(nn,-1)
    softmaxin = nn.output.op.inputs[0]
    camT = gradCam(softmaxin,gradCamA)

# TODO: make qt part work in thread
# TODO: fix this, (bad fast way to exit from programm)
close_main_loop = [False]

def rescale_img(image):
    img = np.uint8((1. - (image - np.min(image)) * 1. / (np.max(image) - np.min(image))) * 255)
    return img

def values2Map(values, num_cols=20):
    size = len(values)
    vals_filled = np.append(values, [0] * ((num_cols - len(values) % num_cols) % num_cols))
    value_map = vals_filled.reshape(-1, num_cols)
    scaled_map = (value_map-value_map.min()) / (value_map.max()-value_map.min())
    img = cv2.applyColorMap(np.uint8(scaled_map*255), cv2.COLORMAP_JET)
    return img, size

def assignWhenChanged(var,value):
    # Assingning variable takes much time
    var_value = sess.run(var)
    if var_value != value:
        print(f" Variable value changed {value}!= {var_value}")
        sess.run(var.assign(value))

async def main(ui=None, options={}):
    assert ui
    ui.fillLayers(conv_grids.keys(), fc_outputs.keys())

    with StreamReader(args.stream) as cap:
        fps = FPS()
        for frame,framenum in zip(cap.read(),itertools.count()):

            if ui.paused:
                frame = old_frame
            else:
                old_frame = frame
            currentGridName = ui.currentConv

            timer = Timer("processing",silent=True)
            ui.loadRealImage(frame)
            timer.tick("image loaded")
            map_raw_idx = ui.convMap.raw_idx
            dense_raw_idx = ui.denseMap.raw_idx

            frame = cv2.resize(frame,(224,224))
            frameToShow = frame.copy()
            frame = np.array([frame])
            timer.tick("frame prepared")
            gridTensor,(columns,rows), mapStack= conv_grids[currentGridName]
            neuronBackpropT,map_neuron_selection_T = convBackprops[currentGridName]
            timer.tick("setting graph vars")
            if map_raw_idx < len(mapStack):
                assignWhenChanged(map_neuron_selection_T, map_raw_idx)

            timer.tick("running main session")
            fetches = {
                "grid": gridTensor,
                "map": mapStack[map_raw_idx],
                # "cam": camT,
                "neuronBackprop": neuronBackpropT,
                # "dense": fc_outputs[currentDense]
            }
            # if the network has fully connected layers their inspection
            # as well as GradCam algorithm is possible
            if fc_outputs:
                currentDense = ui.currentDense
                fetches.update({
                    "dense":  fc_outputs[currentDense],
                    "cam": camT
                    })

            if "dense" in fetches:
                assignWhenChanged(fc_backprops[currentDense].selection, dense_raw_idx)

            fetched = sess.run(fetches,
                                feed_dict={ph:frame})
            timer.tick("Session passed")
            if "cam" in fetched:
                heatmap, coloredMap = gradCamToHeatMap(fetched["cam"],frameToShow)
                cv2.imshow("gradCam",coloredMap)
            if "dense" in fetched:
                activationMap, cell_numbers = values2Map(fetched["dense"][0])
                ui.loadActivationMap(activationMap)
                ui.loadActivationScrollMap(activationMap, cell_numbers)
                if dense_raw_idx < cell_numbers:
                    ui.setDenseValue(fetched["dense"][0][dense_raw_idx])
            if "grid" in fetched:
                ui.loadMap(rescale_img(fetched["grid"]), (rows,columns))
            if "map" in fetched:
                ui.loadCell(rescale_img(fetched["map"]))
            cv2.imshow("neuron-backprop",fetched["neuronBackprop"][0])

            print(f"Frame Number:{framenum:5d}, FPS:{fps():2.1f}", end="\r")
            # cv2.imshow("neuron-backprop-fc",fc_backprop[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # TODO: add check for number of cells here

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
    loop  = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, sigint_handler)

    app = QApplication(sys.argv)
    ui = Ui()
    ui.show()

    loop.run_until_complete(main(ui=ui))

# writer = tf.summary.FileWriter("outputgraph", sess.graph)
# writer.close()
