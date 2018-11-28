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
from deconv import unknownVis, createGrad
from gradCam import gradCam, gradCamToHeatMap
from networks import getNetwork

graph = tf.get_default_graph()
def getLayers(type='Conv2D'):
    return [i for i in graph.get_operations() if i.type.lower() == type.lower()]
    # return [i.type for i in graph.get_operations()]
def getLayersOutputs(type='Conv2D'):
    return [(i.outputs[0],i.name) for i in graph.get_operations() if i.type.lower() == type.lower()]
    # return [i.type for i in graph.get_operations()]




def constructGridNode(output):
    layer = output
    numMaps = int(layer.shape[-1])
    numColumns = math.ceil(numMaps**0.5)
    numRows = math.ceil(numMaps/numColumns)
    zerosNum = numRows*numColumns-numMaps
    zerosShape = [int(i) for i in layer.shape]
    zerosShape[-1] = zerosNum
    zeros = tf.zeros(
        zerosShape,
        dtype=tf.float32,
        name=None)
    concated = tf.concat([layer,zeros],-1)
    len,width,depth= [s for s in concated.shape]
    mapStack =tf.unstack(concated,axis=2)
    rowStacks = [tf.concat(mapStack[i:i+numColumns],axis=1) for i in range(0,numColumns*numRows,numColumns)]
    result = tf.concat(rowStacks,axis=0)
    return result,(numColumns,numRows), mapStack
    # reshaped = tf.reshape(concated,(len*numColumns,width*numRows))
    # return reshaped

sess = tf.Session()
tf.keras.backend.set_session(sess)

nn, ph = getNetwork()
convOutputs = getLayersOutputs()
convLayers = getLayers()
convGrids = {name: constructGridNode(output[0]) for (output,name) in convOutputs}
unknownVisNodes = unknownVis(graph,convGrids,ph)


gradCamA = nn.layers[-6 ].output
softmaxin = nn.output.op.inputs[0]
camT = gradCam(softmaxin,gradCamA)



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


parser = argparse.ArgumentParser()
parser.add_argument('--stream', default="http://192.168.16.101:8081/video",
                    help="Input Video")
parser.add_argument('--show', default=True,
                    help="Show output window")

args = parser.parse_args()



loop  = asyncio.get_event_loop()

def defaultCb(frame):
    # print("showing frame {}".format(fram))
    cv2.imshow('frame',frame)

def illustrateDetections(frame,vggOut):
    preds = tf.keras.applications.vgg16.decode_predictions(vggOut)[0][0][1]
    cv2.putText(frameToShow,"{} {}".format(np.argmax(vggOut,axis=1),preds),(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1,cv2.LINE_AA)


from ui import Ui
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread
import sys

def junkish():
            gradList = unknownVisNodes[currentGridName]
            # if gradList[raw_idx] is None:
            #     gradList[raw_idx] = createGrad(graph,mapStack[raw_idx],ph)
            # print("grad operation",gradList[raw_idx])
            # aGrid, certainMap,reconst = sess.run([gridTensor,mapStack[raw_idx],gradList[raw_idx]],feed_dict={ph:frame})

            aGrid, certainMap,gradCamOut = sess.run([gridTensor,mapStack[raw_idx],gradCamOutT],feed_dict={ph:frame})

            # reconst = reconst[0][0]
            # cv2.imshow("unknownVis",reconst)
            heatmap, coloredMap = gradCamToHeatMap(gradCamOut,frameToShow)
            cv2.imshow("gradCam",coloredMap)
                # cv2.imshow("gradCam",gradCamOut)

            # print(f"aGrid {aGrid.shape}{(columns,rows)}")

            # aGrid = cv2.resize(aGrid, (500,500))
            # defaultCb(aGrid)

            # vggOut = sess.run(vgg.output,feed_dict={imPh:frame})
            # print("{} nn argmax: {} {}".format(time.time(), np.argmax(vggOut,axis=1),preds))
            # if args.show:
            #     # cv2.putText(frame,"Persons: {}".format("3"),(40,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)



async def main(ui=None,cb=defaultCb, options={}):
    assert ui
    ui.setButtons(convGrids.keys())

    with StreamReader(args.stream) as cap:

        for frame in cap.read():
            currentGridName = ui.currentMap
            ui.loadRealImage(frame)
            raw_idx = ui.fmap.raw_idx

            frame = cv2.resize(frame,(224,224))
            frameToShow = frame.copy()
            frame = np.array([frame])
            gridTensor,(columns,rows), mapStack= convGrids[currentGridName]

            aGrid, certainMap,cam = sess.run([gridTensor,mapStack[raw_idx],camT],feed_dict={ph:frame})
            heatmap, coloredMap = gradCamToHeatMap(cam,frameToShow)
            cv2.imshow("gradCam",coloredMap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ui.loadCell(certainMap)
            ui.loadMap(aGrid, (rows,columns))
            QApplication.processEvents()




if __name__ == '__main__':

    app = QApplication(sys.argv)
    ui = Ui()
    ui.show()
    loop.run_until_complete(main(ui=ui))
    sys.exit(app.exec_())
