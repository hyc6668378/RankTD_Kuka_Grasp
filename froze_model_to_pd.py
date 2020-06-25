#-*-coding:utf-8-*-

from alg.ppo.ppo2 import PPO2
import tensorflow as tf

"""
Solidify the model with satisfactory training (.zip) into pd file
"""

Model_Name = 'model/current_best.zip'

model = PPO2.load(Model_Name)

model_pd_name = 'Rank_TD.pd'
with model.graph.as_default():
    graphdef_inf = tf.graph_util.remove_training_nodes(model.graph.as_graph_def())
    graphdef_frozen = tf.graph_util.convert_variables_to_constants(model.sess, graphdef_inf, ['model/deterministic_action'])
    tf.io.write_graph(graphdef_frozen, "model", model_pd_name, as_text=False)
    print("Frozen_graph in 'model/"+model_pd_name)
