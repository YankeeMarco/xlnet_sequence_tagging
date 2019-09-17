# coding=utf-8
import re


def gen_piece(pieces, tokens):
    for pie in zip(pieces, tokens):
        yield pie


def gen_tags4piece(pieces, tokens, list_words, list_tags):
    gen_p = gen_piece(pieces, tokens)
    list_p_tags = []
    for (word, tag) in zip(list_words, list_tags):
        word = word.strip()
        concat_piece = ""
        #print("\"" + word + "\"")
        while concat_piece != word:
            try:
                piece, token = gen_p.next()
            except Exception as _:
                break
            #print("piece: |{}|".format(piece))
            concat_piece += re.sub(r"▁", "", piece)
            if concat_piece == word:
                #print("concat_piece:\"" + concat_piece + "\"")
                list_p_tags.append(tag)
                break
            else:
                list_p_tags.append(tag)
    assert len(list_p_tags) == len(pieces)
    return [0 if i == "O" else 1 for i in list_p_tags]
