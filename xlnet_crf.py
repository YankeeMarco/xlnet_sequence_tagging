# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata
import six
from functools import partial

SPIECE_UNDERLINE = '▁'
from os.path import join
from absl import flags
import os
import re
from tensorflow.python import debug as tf_debug

os.chdir(os.path.expanduser("~") + "/Documents/xlnet_sequence_tagging")
import csv
import collections
import numpy as np
import time
import math
import json
import random
from copy import copy
from collections import defaultdict as dd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import absl.logging as _logging  # pylint: disable=unused-import
# os.environ.setdefault(key='BASE_DIR', value='gs://bucket20190704/xlnet_models/xlnet_cased_L-12_H-768_A-12')
# os.environ.setdefault(key='GS_ROOT', value='gs://bucket20190704/')
# import pydevd_pycharm
# pydevd_pycharm.settrace('127.0.0.1', port=12345, stdoutToServer=True, stderrToServer=True)


import tensorflow as tf
import sentencepiece as spm

from data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
import model_utils
import function_builder
from classifier_utils import PaddingInputExample

ptb_ud_dict = {'#': 'SYM',
               '$': 'SYM',
               "''": 'PUNCT',
               ',': 'PUNCT',
               '-LRB-': 'PUNCT',
               '-RRB-': 'PUNCT',
               '.': 'PUNCT',
               ':': 'PUNCT',
               'AFX': 'ADJ',
               'CC': 'CCONJ',
               'CD': 'NUM',
               'DT': 'DET',
               'EX': 'PRON',
               'FW': 'X',
               'HYPH': 'PUNCT',
               'IN': 'ADP',
               'JJ': 'ADJ',
               'JJR': 'ADJ',
               'JJS': 'ADJ',
               'LS': 'X',
               'MD': 'VERB',
               'NIL': 'X',
               'NN': 'NOUN',
               'NNP': 'PROPN',
               'NNPS': 'PROPN',
               'NNS': 'NOUN',
               'PDT': 'DET',
               'POS': 'PART',
               'PRP': 'PRON',
               'PRP$': 'DET',
               'RB': 'ADV',
               'RBR': 'ADV',
               'RBS': 'ADV',
               'RP': 'ADP',
               'SYM': 'SYM',
               'TO': 'PART',
               'UH': 'INTJ',
               'VB': 'VERB',
               'VBD': 'VERB',
               'VBG': 'VERB',
               'VBN': 'VERB',
               'VBP': 'VERB',
               'VBZ': 'VERB',
               'WDT': 'DET',
               'WP': 'PRON',
               'WP$': 'DET',
               'WRB': 'ADV',
               '``': 'PUNCT'}
uds = ['ADJ',
       'ADP',
       'PUNCT',
       'ADV',
       'AUX',
       'SYM',
       'INTJ',
       'CCONJ',
       'X',
       'NOUN',
       'DET',
       'PROPN',
       'NUM',
       'VERB',
       'PART',
       'PRON',
       'SCONJ']
# Model
flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")
flags.DEFINE_string("summary_type", default="last",
                    help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,
                  help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", default=False,
                  help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,
                  help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="/mnt/disk1/data/xlnet_output_dir",
                    help="Output dir for TF records.")
flags.DEFINE_string("spiece_model_file", default="",
                    help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("data_dir", default="/home/dev/udify-master/data/ud/",
                    help="Directory for input data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
                          "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")

# Training
flags.DEFINE_bool("do_train", default=True, help="whether to do training")
flags.DEFINE_integer("train_steps", default=12000,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=2e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0,
                     help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=3,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=8,
                     help="Batch size for training. Note that batch size 1 corresponds to "
                          "4 sequences: one paragraph + one quesetion + 4 candidate answers.")
flags.DEFINE_float("weight_decay", default=0.00, help="weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# Evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_string("eval_split", default="dev",
                    help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=32,
                     help="Batch size for evaluation.")

# Data config
flags.DEFINE_integer("max_seq_length", default=2048,
                     help="Max length for the paragraph.")
flags.DEFINE_integer("max_qa_length", default=128,
                     help="Max length for the concatenated question and answer.")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")
flags.DEFINE_bool("uncased", default=False,
                  help="Use uncased.")
flags.DEFINE_bool("high_only", default=True,
                  help="Evaluate on high school only.")
flags.DEFINE_bool("middle_only", default=False,
                  help="Evaluate on middle school only.")

FLAGS = flags.FLAGS

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def gen_piece(pieces, tokens):
    for pie in zip(pieces, tokens):
        yield pie


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    return f


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=values))
    return f


def create_null_tfexample():
    # this is for evaluation, padding to batchsize-long
    features = InputFeatures(
        input_ids=[0] * FLAGS.max_seq_length,
        input_mask=[1] * FLAGS.max_seq_length,
        segment_ids=[0] * FLAGS.max_seq_length,
        label_id=[0] * FLAGS.max_seq_length,
        is_real_example=False)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


def gen_sentence(ud_data_dir, set_flag, sp_model):
    """
    processing the text files from conllu format to sentence generator
    :param ud_data_dir: the data dir of train/eval dirs
    :param set_flag: the "train"/"eval" dir flag
    :param sp_model: the sentencepieces model
    :return: dict of tokens, ids, text, tags etc.
    """
    cur_dir = os.path.join(ud_data_dir, set_flag)
    # print("********************  cur_dir is {}".format(cur_dir))
    rawtext = ""
    wordlist = []
    taglist = []
    just_yield = False
    # filename_counter = 0
    for filename in tf.gfile.ListDirectory(cur_dir):
        # if not re.search(r"\.conll", filename):
        #     continue
        # print(filename)
        # filename_counter += 1
        # print(filename_counter)
        cur_path = os.path.join(cur_dir, filename)
        with tf.gfile.Open(cur_path) as f:
            line = f.readline()
            while line:
                if just_yield:
                    rawtext = ""
                    wordlist = []
                    taglist = []
                    just_yield = False
                    # if it has just yield one sentence, read a new line first!
                    continue
                if ".conll" in filename:
                    if len(re.findall(r"\t", line)) > 5 and re.search(r"^\d+", line):
                        ll = [i for i in re.split(r"\t", line)]
                        assert len(ll) == 10
                        word = ll[1]
                        tag = ll[3]
                        if tag not in ["_"]:
                            assert tag in uds
                            assert not re.search(r"\s", word)
                            assert not re.search(r"\s", tag)
                            wordlist.append(word.strip())
                            taglist.append(tag.strip())
                            if not re.match(r"^'s$|^'ll$|^'re$|^n't$|^[,.!?{\-(\[@]$|^'d$", word):
                                # but if pre-word is one of these, the word shoud attached with a back-slice to cut the last space
                                if re.match(r"^[\-$@(\[{]$", wordlist[-1]):
                                    rawtext = "{}".format(rawtext[:-1] + word + " ") if len(rawtext) > 0 else word
                                else:
                                    rawtext = rawtext + word + " "
                            else:
                                rawtext = rawtext[:-1] + word + " "
                if "gold_conll" in filename:
                    if len(re.findall(r"/", re.split(r"\s+", line)[0])) > 2:
                        ll = [i.strip() for i in re.split(r"\s+", line) if len(i) > 0]
                        word = re.sub(r"^/", "", ll[3])
                        tag = ll[4]
                        if tag not in ["XX", "UH", "``", "''", "NFP", "ADD", "*"]:
                            # if tag in ptb_ud_dict.keys():
                            assert tag in ptb_ud_dict.keys()
                            assert not re.search(r"\s", word)
                            wordlist.append(word)
                            taglist.append(ptb_ud_dict[tag])
                            # normal word to attach or concat, eg., 'rich'; firstly, the coming word will be attached withou pre-space
                            if not re.match(r"^'s$|^'ll$|^'re$|^n't$|^[,.!?{\-(\[@]$|^'d$", word):
                                # but if pre-word is one of these, the word shoud attached with a back-slice to cut the last space
                                if re.match(r"^[\-$@(\[{]$", wordlist[-1]):
                                    rawtext = "{}".format(rawtext[:-1] + word + " ") if len(rawtext) > 0 else word
                                else:
                                    rawtext = rawtext + word + " "
                            else:
                                rawtext = rawtext[:-1] + word + " "

                if re.match(r"^\n$", line) and not just_yield and len(taglist) > 0:
                    just_yield = True
                    pieces, tokens = encode_ids(sp_model, rawtext, sample=False)
                    assert len(pieces) == len(tokens)
                    dic_sentence = dict(
                        rawtext=rawtext,
                        wordlist=wordlist,
                        taglist=taglist,
                        pieces=pieces,
                        tokens=tokens
                    )
                    yield dic_sentence
                line = f.readline()


def process_conllu2tfrecord(ud_data_dir, set_flag, tfrecord_path, sp_model):
    if tf.gfile.Exists(tfrecord_path) and not FLAGS.overwrite_data:
        return
    tf.logging.info("Start writing tfrecord %s.", tfrecord_path)
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    eval_batch_example_count = 0
    dic_concat = dict(rawtext="",
                      wordlist=[],
                      taglist=[],
                      pieces=[],
                      tokens=[])
    generator_sen = gen_sentence(ud_data_dir, set_flag, sp_model)
    while True:
        try:
            sentence_dic = generator_sen.next()
        except Exception as _:
            #  drop the last rawtext of ${FLAGS.max_seq_length} tokens ( it is OK or let's fix it later, now focusing on xlnet model)
            break

        if len(sentence_dic['tokens']) < (FLAGS.max_seq_length - 3 - len(dic_concat['tokens'])):
            dic_concat['tokens'].extend(sentence_dic['tokens'])
            dic_concat['pieces'].extend(sentence_dic['pieces'])
            dic_concat['wordlist'].extend(sentence_dic['wordlist'])
            dic_concat['taglist'].extend(sentence_dic['taglist'])
            dic_concat['rawtext'] += sentence_dic['rawtext']
        else:
            pieces = dic_concat['pieces']
            tokens = dic_concat['tokens']
            wordlist = dic_concat['wordlist']
            taglist = dic_concat['taglist']
            p_tag_list = []
            gen_p = gen_piece(pieces, tokens)
            for (word, tag) in zip(wordlist, taglist):
                concat_piece = ""
                # print("\"" + word + "\"")
                while concat_piece != word:
                    try:
                        piece, token = gen_p.next()
                    except Exception as _:
                        break
                    # print("piece: |{}|".format(piece))
                    concat_piece += re.sub(r"▁", "", piece)
                    if concat_piece == word:
                        # print("concat_piece:\"" + concat_piece + "\"")
                        p_tag_list.append(uds.index(tag) + 10)
                        break
                    else:
                        p_tag_list.append(uds.index(tag) + 10)
            assert len(p_tag_list) == len(pieces)
            all_label_id = p_tag_list
            segment_ids = [SEG_ID_A] * len(tokens)
            tokens.append(SEP_ID)
            all_label_id.append(SEP_ID)
            segment_ids.append(SEG_ID_A)

            tokens.append(SEP_ID)
            all_label_id.append(SEP_ID)
            segment_ids.append(SEG_ID_B)

            tokens.append(CLS_ID)
            all_label_id.append(CLS_ID)
            segment_ids.append(SEG_ID_CLS)

            cur_input_ids = tokens
            cur_input_mask = [0] * len(cur_input_ids)
            cur_label_ids = all_label_id
            if len(cur_input_ids) < FLAGS.max_seq_length:
                delta_len = FLAGS.max_seq_length - len(cur_input_ids)
                cur_input_ids = [0] * delta_len + cur_input_ids
                cur_input_mask = [1] * delta_len + cur_input_mask
                segment_ids = [SEG_ID_PAD] * delta_len + segment_ids
                cur_label_ids = [0] * delta_len + cur_label_ids

            assert len(cur_input_ids) == FLAGS.max_seq_length
            assert len(cur_input_mask) == FLAGS.max_seq_length
            assert len(segment_ids) == FLAGS.max_seq_length
            assert len(cur_label_ids) == FLAGS.max_seq_length

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(cur_input_ids)
            features["input_mask"] = create_float_feature(cur_input_mask)
            features["segment_ids"] = create_int_feature(segment_ids)
            features["label_ids"] = create_int_feature(cur_label_ids)
            features["is_real_example"] = create_int_feature([True])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            if set_flag == "eval":
                eval_batch_example_count += 1
            dic_concat = sentence_dic
    if set_flag == "eval" and eval_batch_example_count % FLAGS.eval_batch_size != 0:
        tf_example = create_null_tfexample()
        for i in range(FLAGS.eval_batch_size - eval_batch_example_count % FLAGS.eval_batch_size):
            writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    tf.logging.info("Input tfrecord file {}".format(input_file))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = int()
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = FLAGS.train_batch_size
        elif FLAGS.do_eval:
            batch_size = FLAGS.eval_batch_size

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
            d = d.repeat()
            # d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def get_model_fn():
    def model_fn(features, mode, params):
        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # labels = features['label_ids']
        total_loss, per_example_loss, logits = function_builder.get_ner_loss(
            FLAGS, features, is_training)  # , lengths=lengths)
        print("get model function features :{}".format(features))
        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        #### load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        #### Evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            assert FLAGS.num_hosts == 1

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                eval_input_dict = {
                    'labels': label_ids,
                    'predictions': predictions,
                    'weights': is_real_example
                }
                accuracy = tf.metrics.accuracy(**eval_input_dict)

                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {'eval_accuracy': accuracy, 'eval_loss': loss}

            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

            #### Constucting evaluation TPUEstimatorSpec with new cache.
            label_ids = tf.reshape(features['label_ids'], [-1])
            metric_args = [per_example_loss, label_ids, logits, is_real_example]

            if FLAGS.use_tpu:
                eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=(metric_fn, metric_args),
                    scaffold_fn=scaffold_fn)
            else:
                eval_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=metric_fn(*metric_args))

            return eval_spec

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        #### Constucting training TPUEstimatorSpec with new cache.
        if FLAGS.use_tpu:
            #### Creating host calls
            host_call = None

            train_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
                scaffold_fn=scaffold_fn)
        else:
            train_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    # return_unicode is used only for py2
    # note(zhiliny): in some systems, sentencepiece only accepts str for py2
    if six.PY2 and isinstance(text, unicode):
        text = text.encode('utf-8')

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = piece.decode('utf-8')
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return pieces, ids


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    #### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)

    # def tokenize_fn(text):
    #     text = preprocess_text_ner(text, lower=FLAGS.uncased)
    #     return encode_ids(sp, text)

    # TPU Configuration
    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = get_model_fn()

    spm_basename = os.path.basename(FLAGS.spiece_model_file)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if FLAGS.do_train:
        train_file_path_base = "CRF{}.len-{}.train.tf_record".format(
            spm_basename, FLAGS.max_seq_length)
        train_file_path = os.path.join(FLAGS.output_dir, train_file_path_base)
        if not tf.gfile.Exists(train_file_path) or FLAGS.overwrite_data:
            process_conllu2tfrecord(FLAGS.data_dir, "train", train_file_path, sp)
        # if not tf.gfile.Exists(train_file_path) or FLAGS.overwrite_data:
        #     train_examples = get_examples_ner(FLAGS.data_dir, "train")
        #     random.shuffle(train_examples)
        #     file_based_convert_examples_to_features_ner(
        #         train_examples, tokenize_fn, train_file_path)
        # hook = tf_debug.TensorBoardDebugHook(grpc_debug_server_addresses="localhost:2333")
        # hook = tf_debug.LocalCLIDebugHook(ui_type="readline")
        # hook = tf_debug.LocalCLIDebugHook()
        # hook = tf_debug.GrpcDebugHook()
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file_path,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)  # hooks=[hook])

    if FLAGS.do_eval:
        eval_file_path_base = "CRF{}.len-{}.{}.tf_record".format(
            spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
        eval_file_path = os.path.join(FLAGS.output_dir, eval_file_path_base)
        if not tf.gfile.Exists(eval_file_path) or FLAGS.overwrite_data:
            process_conllu2tfrecord(FLAGS.data_dir, "eval", eval_file_path, sp)
        #
        # eval_examples = get_examples_ner(FLAGS.data_dir, FLAGS.eval_split)
        # tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on. These do NOT count towards the metric (all tf.metrics
        # support a per-instance weight, and these get a weight of 0.0).
        #
        # Modified in XL: We also adopt the same mechanism for GPUs.

        # while len(eval_examples) % FLAGS.eval_batch_size != 0:
        #     eval_examples.append(PaddingInputExample())

        # if FLAGS.high_only:
        #     eval_file_path_base = "high." + eval_file_path_base
        # elif FLAGS.middle_only:
        #     eval_file_path_base = "middle." + eval_file_path_base

        # eval_file_path = os.path.join(FLAGS.output_dir, eval_file_path_base)
        # file_based_convert_examples_to_features_ner(
        #     eval_examples, tokenize_fn, eval_file_path)

        # assert len(eval_examples) % FLAGS.eval_batch_size == 0
        # eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)
        eval_steps = 8

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file_path,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True)

        ret = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=eval_steps)

        # Log current result
        tf.logging.info("=" * 80)
        log_str = "Eval | "
        for key, val in ret.items():
            log_str += "{} {} | ".format(key, val)
        tf.logging.info(log_str)
        tf.logging.info("=" * 80)


if __name__ == "__main__":
    tf.app.run()
