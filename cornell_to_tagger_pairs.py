# coding=utf-8
import re

# wf_path = "/home/dev/udify-master/data/ud/test.en"
wf_path = "/home/mm/Documents/udify-master/data/test256.en"
rf_path = "/home/mm/Documents/udify-master/data/ud-treebanks-v2.3/UD_English-ParTUT/en_partut-ud-train.conllu"
wf = open(wf_path, "a")
with open(rf_path, "r") as rf:
    import re
    line = rf.readline()
    empty_line_wrote = False
    to_write_len = 0
    last_sentence = ''
    while line:
        # line = rf.readline()
        if re.match(r"^\d", line):
            empty_line_wrote = False
            ll = re.split(r"\s+", line)
            word = ll[1]
            assert not re.search(r"\s", word)
            # tag = 'NOUN' if ll[3] == 'NOUN' else 'O'
            tag = ll[3]
            last_sentence = last_sentence + word + "\t" + tag + "\n"
            # wf.write(word + "\t" + tag + "\n")
            to_write_len += 1
        elif re.match(r"^#", line) and empty_line_wrote is False and to_write_len > 256:
            wf.write(last_sentence)
            print(last_sentence)
            wf.write("\n")
            to_write_len = 0
            last_sentence = ""
            empty_line_wrote = True
        line = rf.readline()
