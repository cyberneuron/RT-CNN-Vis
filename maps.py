import tensorflow as tf
import math

def mapsToGrid(output):
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
