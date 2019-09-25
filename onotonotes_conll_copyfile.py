# coding=utf-8
import re
import os
import random
from shutil import copyfile

gen_dir = os.walk(top="/home/mm/Documents/OntoNotes-5.0-NER-BIO-master/", topdown=True)
path_list = []
for i in gen_dir:
    for pa_list in i:
        for pa in pa_list:
            if not pa.endswith("gold_conll"):
                continue
            path = os.path.join(i[0], pa)
            if not path.endswith("gold_conll"):
                continue
            with open(path, "r") as rf:
                tenlines = ""
                for l in range(10):
                    tenlines += rf.readline()
                if re.search(r"XX", tenlines):
                    break
                else:
                    path_list.append(path)


trainfiles = random.sample(path_list, 7000)
evalfiles = [i for i in path_list if i not in trainfiles]
for i in trainfiles:
    copyfile(i, "/home/mm/Documents/xlnet_models/data_dir/train/" + "_".join(i.split(r"/")[-4:]))
for i in evalfiles:
    copyfile(i, "/home/mm/Documents/xlnet_models/data_dir/eval/" + "_".join(i.split(r"/")[-4:]))
