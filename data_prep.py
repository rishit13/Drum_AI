import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset
SOS_token  = 0
EOS_token = 1
padding = 2

final_input = []
final_target = []

def notes_to_index_fn(notes_list, pp):
  return [pp.x_note_to_int[key] for key in notes_list]

def int_tokens(input_new, target_new,pp):
  for  input_row,target_row in zip(input_new,target_new):
    changed_row = notes_to_index_fn(input_row,pp)
    changed_tar_row = notes_to_index_fn(target_row,pp)
    changed_row.insert(0, SOS_token)
    changed_tar_row.insert(0, SOS_token)
    changed_row.append(EOS_token)
    changed_tar_row.append(EOS_token)
    final_input.append(changed_row)
    final_target.append(changed_tar_row)
  return final_input,final_target


def data_batch_load(final_input, final_target,pp, batch_size = 32, split = 0.8):
    input_tensor = [torch.LongTensor(b) for b in final_input]
    input_padded = pad_sequence(input_tensor, batch_first = True, padding_value = 3)
    x = input_padded.tolist()
    input_tensor = np.array(x)
    target_tensor = [torch.LongTensor(b) for b in final_target]
    target_padded = pad_sequence(target_tensor, batch_first = True, padding_value = 3)
    x_target = target_padded.tolist()
    target_tensor = np.array(x_target)
    vocab_size = len(pp.x_note_to_int)
    x_tensor = torch.from_numpy(input_tensor).int()
    y_tensor = torch.from_numpy(target_tensor).int()
    dataset = TensorDataset(x_tensor, y_tensor)
    train_len = int(split*len(dataset))
    valid_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size)
    return train_dataset, val_dataset, train_loader, val_loader, vocab_size
