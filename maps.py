import tensorflow as tf
import math

def mapsToGrid(output):
    layer = output
    numMaps = int(layer.shape[-1])
    numColumns = math.ceil(numMaps**0.5)
    numRows = math.ceil(numMaps/numColumns)
    zerosNum = numRows*numColumns-numMaps
    mapStack =tf.unstack(layer,axis=2)
    map_on_tail = mapStack[-1]
    for i in range(zerosNum):
        mapStack.append(map_on_tail)
    rowStacks = [tf.concat(mapStack[i:i+numColumns],axis=1) for i in range(0,numColumns*numRows,numColumns)]
    result = tf.concat(rowStacks,axis=0)
    return result,(numColumns,numRows), mapStack
