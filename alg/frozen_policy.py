#coding=utf-8

import numpy as np
import tensorflow as tf
import os
config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

def load_graph(frozen_graph_filename='model/frozen_model.pb', graph_name="Embedding"):
    # We parse the graph_def file
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=graph_name,
            op_dict=None,
            producer_op_list=None
        )
    return graph

class RankTD_policy:
    def __init__(self):
        frozen_graph_filename = 'model/Rank_TD.pd'
        assert os.path.isfile(frozen_graph_filename), "Can't find {}".format(frozen_graph_filename)

        self._frozen_graph = load_graph(frozen_graph_filename=frozen_graph_filename, graph_name='Rank_TD')

        self.img_placeholder = self._frozen_graph.get_tensor_by_name(self._frozen_graph.get_operations()[0].values()[0].name)
        self.deterministic_a = self._frozen_graph.get_tensor_by_name(self._frozen_graph.get_operations()[-1].values()[-1].name)

    def __call__(self, o):
        img = self.__check_shape(o)
        with tf.compat.v1.Session(graph=self._frozen_graph, config=config) as sess:
            pi = sess.run(self.deterministic_a, feed_dict={ self.img_placeholder: img[np.newaxis, ]})
        return pi[0]

    @staticmethod
    def __check_shape(o):
        assert isinstance(o, np.ndarray), "Observation must be an numpy.ndarray"
        assert o.shape == (256,128,3),  "The dim of observation must be (256,128,3)"
        return o