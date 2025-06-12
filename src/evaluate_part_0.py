""" Usage:
    <file-name> --gold=GOLD_FILE --pred=PRED_FILE [--debug]

Options:
  --help                           Show this message and exit
  -i INPUT_FILE --in=INPUT_FILE    Input file
                                   [default: infile.tmp]
  -o INPUT_FILE --out=OUTPUT_FILE  Input file
                                   [default: outfile.tmp]
  --debug                          Whether to debug
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import sklearn
import itertools
import numpy as np
from sklearn.metrics import f1_score

# Local imports


#----


def flatten(ls):
    """
    flatten a nested list
    """
    flat_ls = list(itertools.chain.from_iterable(ls))
    return flat_ls

class Encode_Multi_Hot:
    """
    change the variable length format into a
    fixed size one hot vector per each label
    """
    def __init__(self):
        """
        init data structures
        """
        self.label_to_ind = {}
        self.ind_to_label = {}
        self.num_of_label = None

    def fit(self, raw_labels):
        """
        learn about possible labels
        """
        # get a list of unique values in df
        labs = list(set(flatten(raw_labels)))
        inds = list(range(len(labs)))
        self.label_to_ind = dict(zip(labs, inds))
        self.ind_to_label = dict(zip(inds, labs))
        self.num_of_label = len(labs)

    def enc(self, raw_label):
        """
        encode variable length category list into multiple hot
        """
        multi_hot = np.zeros(self.num_of_label)
        for lab in raw_label:
            cur_ind = self.label_to_ind[lab]
            multi_hot[cur_ind] = 1
        return multi_hot


def parse_df_labels(df):
    """
    Return a dictionary of response name and values from df
    """
    assert(len(df.columns) == 1)
    resp = df.columns[0]
    ls = [eval(val) for val in df[resp]]
    ret_dict = {"resp": resp, "vals": ls}
    return ret_dict

if __name__ == "__main__":

    # Parse command line arguments
    args = docopt(__doc__)
    gold_fn = Path(args["--gold"])
    pred_fn = Path(args["--pred"])

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Start computation
    gold_labels = parse_df_labels(pd.read_csv(gold_fn, keep_default_na = False))
    pred_labels = parse_df_labels(pd.read_csv(pred_fn, keep_default_na = False))

    # make sure that the same label is annotated in pred and gold
    assert(gold_labels["resp"] == pred_labels["resp"])
    enc = Encode_Multi_Hot()
    gold_vals = gold_labels["vals"]
    pred_vals = pred_labels["vals"]

    # make sure pred and gold annotate the same # of features
    assert(len(gold_vals) == len(pred_vals))
    enc.fit(gold_vals + pred_vals)


    gold_multi_hot = np.array([enc.enc(val) for val in gold_vals])
    pred_multi_hot = np.array([enc.enc(val) for val in pred_vals])

    # Print micro-macro f1
    macro_f1 = f1_score(y_true = gold_multi_hot,
                        y_pred = pred_multi_hot,
                        average = "macro")

    micro_f1 = f1_score(y_true = gold_multi_hot,
                        y_pred = pred_multi_hot,
                        average = "micro")

    logging.info(f"Micro f1 = {micro_f1} \n Macro f1 = {macro_f1}")

    # End
    logging.info("DONE")
