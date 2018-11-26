# deconv gradients
from tensorflow.python.ops import gen_nn_ops
import tensorflow as tf
# def _register_custom_gradients():
#     """
#     Register Custom Gradients.
#     """
#     global is_Registered
#
#     if not is_Registered:
#         # register LRN gradients
@tf.RegisterGradient("Customlrn")
def _CustomlrnGrad(op, grad):
    return grad

# register Relu gradients
@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

def unknownVis(graph,layers,input):
    ret = {}
    with graph.gradient_override_map({'Relu': 'GuidedRelu', 'LRN': 'Customlrn'}):
        for name,other in layers.items():
            mapStack = other[-1]
            print(f"registering gradeints for {name}")
            gradients = [None for map in mapStack]
            print(gradients)
            print("gradients registered")
            ret[name] = gradients
    return ret

def createGrad(graph,map,input):
    print("entering with")
    with graph.gradient_override_map({'Relu': 'GuidedRelu', 'LRN': 'Customlrn'}):
        print("creating grad node")
        grad = tf.gradients(map, input)
    return grad
