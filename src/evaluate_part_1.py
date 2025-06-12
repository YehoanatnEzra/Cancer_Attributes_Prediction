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
from sklearn.metrics import mean_squared_error

# Local imports


#----


def parse_df_labels(df):
    """
    Return a dictionary of response name and values from df
    """
    assert(len(df.columns) == 1)
    resp = df.columns[0]
    ls = [float(val) for val in df[resp]]
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
    assert(gold_labels["resp"] == pred_labels["resp"])
    gold_vals = gold_labels["vals"]
    pred_vals = pred_labels["vals"]

    # Compute trivial performance
    trivial_val = np.mean(gold_vals)
    trivial_vals = [trivial_val for _ in gold_vals]
    
    assert(len(gold_vals) == len(pred_vals))


    # Print mse
    mse = mean_squared_error(y_true = gold_vals,
                             y_pred = pred_vals)

    logging.info(f"MSE = {mse}")

    # Trivial performance for reference
    trivial_mse = mean_squared_error(y_true = gold_vals,
                                     y_pred = trivial_vals)
    logging.info(f"For reference, trivial (mean) mse: {trivial_mse}")

    # End
    logging.info("DONE")
