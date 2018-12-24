import tensorflow as tf
import cv2
import numpy as np
import os
import argparse
from streamReader import StreamReader
import asyncio
import time
import math
from collections import namedtuple, OrderedDict
from gradCam import gradCam, gradCamToHeatMap
from guidedBackprop import registerConvBackprops,register_fc_backprops
from networks import getNetwork
from maps import mapsToGrid

parser = argparse.ArgumentParser()
parser.add_argument('--stream', default="http://192.168.16.101:8081/video",
# parser.add_argument('--stream', default="http://191.167.15.101:8079",
                    help="Video stram URI, path to video or webcam number based on which the network is visualized")
# parser.add_argument('--show', default=True,
#                     help="Show output window")
parser.add_argument('--network', default="VGG16",
                    help="Network to visualise (VGG16,ResNet50 ...)")


def get_outputs_from_graph(type='Conv2D'):
    assert type in ['Conv2D']
    return OrderedDict((i.name, i.outputs[0]) for i in graph.get_operations() if i.type.lower() == type.lower())

def get_outputs_from_model(layer_type="Dense",pre_eactivation=True):
    assert layer_type in ["Dense"]
    Layer = getattr(tf.keras.layers,layer_type)
    layers = nn.layers
    def get_layer_output(layer):
        if pre_eactivation:
            return layer.output.op.inputs[0]
        else:
            return layer.output
    # Outputs = namedtuple(type+"Outputs",[layer.name if type(layer) is Layer])
    return OrderedDict( (layer.name, get_layer_output(layer) ) for layer in layers if type(layer) is Layer)


def getLastConv():
    layers = nn.layers
    return [layer.output for layer in layers if type(layer) is tf.keras.layers.Conv2D][-2]



args = parser.parse_args()
graph = tf.get_default_graph()
sess = tf.Session()
sess.as_default()

tf.keras.backend.set_session(sess)
network_name = args.network
# network_name = "MobileNet"

nn, ph = getNetwork(network_name)
print(nn.summary())

convOutputs = get_outputs_from_graph(type='Conv2D')



convGrids = OrderedDict( (name, mapsToGrid(output[0])) for name, output in convOutputs.items())



convBackprops = registerConvBackprops(convOutputs,nn.input)

fc_outputs = get_outputs_from_model(layer_type="Dense")
fc_backprops = register_fc_backprops(fc_outputs,nn.input)

sess.run(tf.variables_initializer([convBackprops[name][1] for name in convBackprops ]))

gradCamA = getLastConv()
gradCamA
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

async def main(ui=None, options={}):
    assert ui
    ui.fillLayers(convGrids.keys(), fc_outputs.keys())

    with StreamReader(args.stream) as cap:

        for frame in cap.read():

            if ui.paused:
                frame = old_frame
            else:
                old_frame = frame
            currentGridName = ui.currentConv
            currentDense = ui.currentDense
            ui.loadRealImage(frame)
            map_raw_idx = ui.convMap.raw_idx
            dense_raw_idx = ui.denseMap.raw_idx

            frame = cv2.resize(frame,(224,224))
            frameToShow = frame.copy()
            frame = np.array([frame])
            gridTensor,(columns,rows), mapStack= convGrids[currentGridName]
            neuronBackpropT,map_neuron_selection_T = convBackprops[currentGridName]
            if map_raw_idx < len(mapStack):
                sess.run(map_neuron_selection_T.assign(map_raw_idx))

            sess.run(fc_backprops[currentDense].selection.assign(dense_raw_idx))
            sess_run = sess.run([gridTensor,mapStack[map_raw_idx],
                                 camT,neuronBackpropT, fc_outputs[currentDense],fc_backprops[currentDense].output],
                                feed_dict={ph:frame})
            aGrid, certainMap, cam, neuronBackprop, denseActs,fc_backprop = sess_run
            heatmap, coloredMap = gradCamToHeatMap(cam,frameToShow)
            activationMap, cell_numbers = values2Map(denseActs[0])
            cv2.imshow("gradCam",coloredMap)
            cv2.imshow("neuron-backprop",neuronBackprop[0])
            print(fc_backprop.shape)
            cv2.imshow("neuron-backprop-fc",fc_backprop[0])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ui.loadActivationMap(activationMap)
            ui.loadActivationScrollMap(activationMap, cell_numbers)
            # TODO: add check for number of cells here
            ui.loadCell(rescale_img(certainMap))
            ui.loadMap(rescale_img(aGrid), (rows,columns))
            if dense_raw_idx < cell_numbers:
                ui.setDenseValue(denseActs[0][dense_raw_idx])

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
