import numpy as np
import random

"""

"""

SOS_token  = 0
EOS_token = 1
padding = 2

class Preproc:
    def __init__(self, input_path, target_path):
        self.input_path = input_path
        self.target_path = target_path
        self.empty_list = []  # stores the indices of inputs that have less than 150 sequence token
        self.clean_input = []
        self.clean_target = []
        self.notes_list_in = []  # list of  all notes in input set
        self.notes_list_target = []  # list of all notes in target set
        self.unique_list = []  # list of all unique notes in the dataset

    def read_files(self):
        with open(self.input_path, 'rb') as file:  # read the pickled input file
            self.input_list = np.load(file, allow_pickle=True)
        with open(self.target_path, 'rb') as target_file:
            self.target_list = np.load(target_file, allow_pickle=True)  # read the pickled output file
        self.input_list_1 = self.input_list.tolist()
        self.target_list_1 = self.target_list.tolist()

    def data_prep(self, limit_length):
        for index, (arr_in, arr_tar) in enumerate(zip(self.input_list_1, self.target_list_1)):
            if (len(arr_in) > limit_length or len(arr_tar) == 0 or len(arr_in) == 0):
                self.empty_list.append(index)
        for index, (arr_in, arr_tar) in enumerate(zip(self.input_list_1, self.target_list_1)):
            if index in self.empty_list:
                continue
            else:
                self.clean_input.append(arr_in)
                self.clean_target.append(arr_tar)
        self.notes_list_in = [element for note_ in self.input_list_1 for element in note_]
        self.notes_list_target = [element for note_ in self.target_list_1 for element in note_]
        result_list = self.notes_list_in + self.notes_list_target
        self.unique_list = set(result_list)
        self.MAX_LEN_input = max(len(input_row) for input_row in self.clean_input)
        self.MAX_LEN_output = max(len(target_row) for target_row in self.clean_target)
        self.x_note_to_int = dict(
            (note_, number) for number, note_ in enumerate(self.unique_list))  # dictionary for note to integer
        self.x_int_to_note = dict(
            (number + 3, note_) for number, note_ in enumerate(self.unique_list))  # dictionary for integer to notes
        for key, value in self.x_note_to_int.items():
            self.x_note_to_int[key] = value + 3
        self.x_note_to_int['<SOS>'] = SOS_token
        self.x_note_to_int['<EOS>'] = EOS_token
        self.x_note_to_int['<PAD>'] = padding
        self.x_int_to_note[padding] = '<PAD>'
        self.x_int_to_note[EOS_token] = '<EOS>'
        self.x_int_to_note[SOS_token] = '<SOS>'
        self.vocab_size = len(self.unique_list ) + 3
