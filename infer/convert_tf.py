import onnx

from onnx_tf.backend import prepare

onnx_model = onnx.load("./gruc_1k6h_torch.onnx")  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("gruc_1k6h_torch2tf")  # export the model
