import numpy as np
import random
import torch
from Generate_drum_beats import generate_beats


def checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss):
  state = {'model' : model, 'best_loss' : best_loss, 'epoch' : epoch, 'rng_state' : torch.get_rng_state(), 'optimizer' : optimizer.state_dict(),}
  torch.save(state, '/checkpoint-NMT')
  torch.save(model.state_dict(), '/checkpoint-NMT-SD')

def train_model(model, train_loader, criterion, epochs, epoch_loss, learning_rate, optimizer,trial_sentence,best_epoch, best_loss,device,pp,save_model = True):

  for epoch in range(epochs):

    print("Epoch - {} / {}".format(epoch+1, epochs))
    model.eval()
    model.train(True)
    tl_sentence = generate_beats(model,trial_sentence,pp.x_note_to_int, pp.x_int_to_note, device)
    for x_batch, y_batch in train_loader:

      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      x_batch = torch.transpose(x_batch,0,1)
      y_batch = torch.transpose(y_batch,0,1)
      output = model(x_batch, y_batch)
      output = output[1:].reshape(-1, output.shape[2])
      y_batch = y_batch[1:].reshape(-1)
      optimizer.zero_grad()
      y_batch = y_batch.type(torch.LongTensor).to(device)
      loss = criterion(output, y_batch)
      loss.backward()
      torch.nn.utils.clip_grad_norm(model.parameters(), max_norm = 1)
      optimizer.step()
      epoch_loss += loss.item()

    #if epoch_loss < best_loss:
      #best_loss = epoch_loss
      #best_epoch = epoch
      #checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss)
      #if ((epoch - best_epoch) >= 4):
       # print("no improvement")
        #break
    print("Epoch_loss = {}".format(loss.item()))
  print(epoch_loss / len(train_loader))
  if (save_model == True):
    torch.save(model,'model_ddb.pth')
    torch.save(model.state_dict(),'model_ddb_weights.pth')