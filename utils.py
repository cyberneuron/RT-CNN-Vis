import tensorflow as tf
from collections import OrderedDict

def get_outputs_from_graph(type='Conv2D'):
    assert type in ['Conv2D']
    graph = tf.get_default_graph()
    return OrderedDict((i.name, i.outputs[0]) for i in graph.get_operations() if i.type.lower() == type.lower())

def get_outputs_from_model(model,layer_type="Dense",pre_eactivation=True):
    assert layer_type in ["Dense","Conv2D"]
    Layer = getattr(tf.keras.layers,layer_type)
    layers = model.layers
    def get_layer_output(layer):
        if pre_eactivation:
            return layer.output.op.inputs[0]
        else:
            return layer.output
    # Outputs = namedtuple(type+"Outputs",[layer.name if type(layer) is Layer])
    return OrderedDict( (layer.name, get_layer_output(layer) ) for layer in layers if type(layer) is Layer)

def getConvOutput(model,index=-1):
    layers = model.layers
    return [layer.output for layer in layers if type(layer) is tf.keras.layers.Conv2D][index]
