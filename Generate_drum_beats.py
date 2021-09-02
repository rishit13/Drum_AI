import torch
import constants
SOS_token = 0
EOS_token = 1
padding   = 2

def notes_to_index_infer(notes_list, x_note_to_int):
  return [x_note_to_int[key] for key in notes_list]

def preprocess_input(input_test, notes_to_index):
  input_test = notes_to_index_infer(input_test, notes_to_index)
  input_test.insert(0, SOS_token)
  input_test.append(EOS_token)
  sentence_tensor = torch.LongTensor(input_test)
  return sentence_tensor


def generate_beats(model,beats_tensor,x_note_to_int, x_int_to_note, device):
  beats_tensor = beats_tensor.unsqueeze(1).to(device)
  with torch.no_grad():
    hidden, cell = model.Encoder_LSTM(beats_tensor)
  outputs = [x_note_to_int.get('<SOS>')]
  for _ in range(constants.MAX_LEN_input):
    previous_word = torch.LongTensor([outputs[-1]]).to(device)
    with torch.no_grad():
      output, hidden,cell = model.Decoder_LSTM(previous_word, hidden, cell)
      best_guess = output.argmax(1).item()
    outputs.append(best_guess)
    if output.argmax(1).item() == x_note_to_int.get('<EOS>'):
      break
  beats_gen = [x_int_to_note[idx] for idx in outputs]
  print(outputs)
  return beats_gen[1:]