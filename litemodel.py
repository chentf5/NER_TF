import tensorflow as tf
#path="./save/saved_model.pb"        #pb文件位置和文件名
#inputs=["inputs"]               #模型文件的输入节点名称
#classes=["classes"]            #模型文件的输出节点名称
#converter = tf.contrib.lite.TocoConverter.from_frozen_graph(path, inputs, classes)
#tflite_model=converter.convert()
#open("./save/model_pb.tflite", "wb").write(tflite_model)

import tensorflow as tf
import os
 
model_dir = './save/'
model_name = 'saved_model.pb'
 
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(sess.graph_def, graph_dir, 'graph.pbtxt',as_text=False)

 
create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')