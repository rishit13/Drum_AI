import random
import torch
import torch.nn as nn

"""
In this script we design the sequence to sequence architecture consisting of the Encoder LSTM class
Decoder LSTM class and the interface class between the Encoder and decoder. 
"""


class EncoderLSTM(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
    super(EncoderLSTM, self).__init__()
    self.input_size = input_size #here the input side is the number of words in the vocabulary, this is used to form the one hot encoding matrix for embeddings
    self.embedding_size = embedding_size # 300
    self.hidden_size = hidden_size #1024
    self.num_layers = num_layers # 2 layers of LSTM
    self.dropout = nn.Dropout(p) # 0.5
    self.tag = True
    self.embedding = nn.Embedding(self.input_size, embedding_size) # shape(755,300)
    self.LSTM = nn.LSTM(self.embedding_size, hidden_size, num_layers, dropout = p) # shape(300, 2, 1024)

  def forward(self,x): # x of shape (seq_length, batch_size) (150, 32)
    #functional calling
    embedding = self.dropout(self.embedding(x)) # shape(150,32,300)
    outputs, (hidden, cell_state) = self.LSTM(embedding) # outputs --> (26, 32,1024) hidden -->(2,32,1024) cell_state = (2,32,1024)
    return hidden, cell_state



"""
For the decoder we use the teacher force approach where we would intially pass the start of sentence token and then 
predict the next word, if the probability of that word is greater than 0.5 then we would use that word as an input 
else we would feed in out target input. 
"""

class DecoderLSTM(nn.Module):

  def __init__(self, input_size,embedding_size,hidden_size, num_layers, p, output_size,vocab_size, device):
    super(DecoderLSTM, self).__init__()
    self.device = device
    self.vocab_size = vocab_size
    self.input_size = input_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.output_size = output_size #the size is the vocab size as there is just one consolodated vocab for guitar and drum notes.
    self.dropout = nn.Dropout(p)
    self.tag = True
    self.embedding = nn.Embedding(self.input_size, self.embedding_size) # shape ( 755, 300)
    self.LSTM = nn.LSTM(self.embedding_size, hidden_size, num_layers, dropout = p) # shape(300, 2, 1024)
    self.fc = nn.Linear(self.hidden_size, self.output_size) # shape ( 1024, 755)

  def forward(self, x, hidden_state, cell_state):
    x= x.unsqueeze(0) #(1,32)
    embedding = self.dropout(self.embedding(x)) #(1,32,300)
    outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state)) # outputs (1,32,1024) , hidden -->(2,32,1024) cell_state = (2,32,1024)
    predictions = self.fc(outputs) # shape ==> (1,32,755)
    predictions = predictions.squeeze(0) # shape (32,755)

    return predictions, hidden_state, cell_state

class Seq2Seq(nn.Module):

  def __init__(self, Encoder_LSTM, Decoder_LSTM, vocab_size,device):
    super(Seq2Seq, self).__init__()
    self.Encoder_LSTM = Encoder_LSTM
    self.Decoder_LSTM = Decoder_LSTM
    self.vocab_size = vocab_size
    self.device = device
  def forward(self, source, target, tfr = 0.5):

    batch_size = source.shape[1] # source shape --> (150,32)
    target_len =target.shape[0] # (832, 32)
    target_vocab_size = self.vocab_size

    outputs = torch.zeros(target_len,batch_size, target_vocab_size).to(self.device)
    hidden_state_encoder, cell_state_encoder = self.Encoder_LSTM(source)
    x = target[0] # Trigger token <SOS>
    for i in range(1, 150):
      output, hidden_state_decoder, cell_state_decoder = self.Decoder_LSTM(x,hidden_state_encoder,cell_state_encoder)
      outputs[i] = output
      best_guess = output.argmax(1)
      x = target[i] if random.random() < tfr else best_guess

    return outputs # shape --> (832, 32,755)

