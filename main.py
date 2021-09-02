from builtins import input
import pickle
from Preprocess_utils import Preproc
from train import train_model
from Music_utils import *
from Generate_drum_beats import preprocess_input, generate_beats
from data_prep import *
from Seq_Model import *
import constants
import sys

"""
The main script contains 2 execution modes, the train mode and inference mode 
The train mode inclusion of 2 file paths (numpy arrays of inputs and output notes).
"""

def main():
    #args = sys.argv[0:]
    train = input("Enter True if you want to train else False : ")#args[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('index_to_notes.pickle', 'rb') as handle: # loading dictionary containing integer to notes translation
        index_to_notes = pickle.load(handle)
    with open('notes_to_index.pickle', 'rb') as handle: # loading dictionary containing notes to integer translation
        notes_to_index = pickle.load(handle)

    if train == "True":
        path1 = input("Enter the input array path : ")
        path2 = input("Enter the target array path : ")
        pp = Preproc(path1, path2) # declaring the input and target pre-processing object
        pp.read_files()
        pp.data_prep(constants.limit_length)
        input_in, target = int_tokens(pp.clean_input, pp.clean_target,pp)
        train_dataset, val_dataset, train_loader, val_loader, vocab_size = data_batch_load(input_in, target,pp) #using pytorch dataloaders to form batch of data using the data_batch_load function present in data_prep script
        trial_sentence = train_dataset[0][0]
        encoder_lstm = EncoderLSTM(constants.input_size, constants.encoder_embeddings_size, constants.hidden_size,
                                   constants.num_layers, constants.encoder_dropout).to(device)
        decoder_lstm = DecoderLSTM(constants.input_size_decoder, constants.decoder_embedding_size,
                                   constants.hidden_size,
                                   constants.num_layers, constants.decoder_dropout,
                                   constants.output_size, constants.vocab_size, device).to(device)
        model = Seq2Seq(encoder_lstm, decoder_lstm,pp.vocab_size,device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=constants.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=constants.pad_index)
        train_model(model, train_loader,criterion, constants.epochs, constants.epoch_loss, constants.learning_rate, optimizer, trial_sentence, constants.best_epoch,
                    constants.best_loss,device,pp, False)
    else:
        file = input("Enter the file to generate drum beats : ")
        file_path = file + ".mid"
        model = torch.load('model_ddb.pth', map_location = torch.device('cpu')) #load the trained model
        model.load_state_dict(torch.load('model_ddb_weights.pth', map_location = torch.device('cpu'))) #load the weights of the model
        input_list = get_notes(file_path) # convert the midi file to numpy array of notes
        input_list_1 = input_list.tolist()
        input_tensor = preprocess_input(input_list_1, notes_to_index) # convert the array into pytorch tensor
        beats = generate_beats(model, input_tensor, notes_to_index, index_to_notes, device) # generate drum beats using Seq2Seq model
        convert_to_midi(beats,file)

if __name__ == "__main__":
  main()






