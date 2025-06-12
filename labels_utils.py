
import numpy as np

class LabelEncoder:
    def __init__(self):
        self.label_to_ind = {}
        self.ind_to_label = {}
        self.num_labels = 0

    def initialize(self, labels_df):
        col_name = labels_df.columns[0]
        all_labels = [eval(val) for val in labels_df[col_name]]
        unique_labels = sorted(set(label for sub in all_labels for label in sub))
        self.label_to_ind = {lab: i for i, lab in enumerate(unique_labels)}
        self.ind_to_label = {i: lab for lab, i in self.label_to_ind.items()}
        self.num_labels = len(unique_labels)

    def to_binary_vector(self, y):
        vector = np.zeros(self.num_labels)
        for lab in y:
            vector[self.label_to_ind[lab]] = 1
        return vector

    def from_binary_vector(self, vec):
        return [self.ind_to_label[i] for i in range(self.num_labels) if vec[i] == 1]
