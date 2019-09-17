# coding=utf-8
# import pandas as pd
import re

# for file in ["train", "test", "dev"]:
#     rf = open("/home/dev/udify-master/data/ud/multilingual/{}.conllu".format(file), "r")
#     wf = open("/home/dev/udify-master/data/ud/multilingual/{}.x.y".format(file), "a")
#     lines_tups = list()
#     oneline = []
#     line = rf.readline()
#     while line:
#         if re.match(r"^\d", line):
#             if re.match(r"^1\t", line):
#                 wf.write("\n")
#             listline = re.split(r"\t", line)
#             wf.write("{}\t1\n".format(listline[1]) if listline[3] == "NOUN" else "{}\t0\n".format(listline[1]))
#         line = rf.readline()

import kashgari
from kashgari.tasks.labeling import BiGRU_CRF_Model
from kashgari.embeddings import BERTEmbedding
from collections import defaultdict


def file2data(file):
    data_x = []
    data_y = []
    _x = []
    _y = []
    line = file.readline()
    while line:
        if line == "\n":
            data_x.append(_x)
            data_y.append(_y)
            _x = []
            _y = []
        else:
            x, y = re.split(r"\t", line)
            _x.append(x)
            _y.append("o" if y == "" else y)
        line = file.readline()
    return data_x, data_y


data_dict = defaultdict(dict)
for file in ["train", "test", "dev"]:
    rf = open("/home/dev/udify-master/data/ud/multilingual/{}.x.y".format(file), "r")
    data_dict[file]["x"], data_dict[file]["y"] = file2data(rf)

bert_embed = BERTEmbedding('/home/dev/bert-weights/multi_cased_L-12_H-768_A-12',
                           task=kashgari.LABELING,
                           sequence_length=100)
model = BiGRU_CRF_Model(bert_embed)
model.fit(x_train=data_dict['train']['x'],
          y_train=data_dict['train']['y'],
          x_validate=data_dict['dev']['x'],
          y_validate=data_dict['dev']['y'],
          epochs=40)
model.save("/home/dev/bert-weights/bert_GRU_Crf.model")
