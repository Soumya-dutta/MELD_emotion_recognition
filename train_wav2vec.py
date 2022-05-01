import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import librosa
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score

#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

np.random.seed(1234)
torch.manual_seed(1234)

#CUDA devices enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

class MyDataset(Dataset):
    '''Dataset class for audio which reads in the audio signals and prepares them for
    training. In particular it pads all of them to the maximum length of the audio
    signal that is present. All audio signals are padded/sampled upto 34s in this
    dataset.
    '''
    def __init__(self, folder, target_dict):
        self.folder = folder
        self.target = target_dict
        wav_files = os.listdir(folder)
        wav_files = [x for x in wav_files if ".wav" in x]
        self.wav_files = wav_files
        self.sr = 16000
        self.duration = 10000

    def __len__(self):
        return len(self.wav_files) 
        
    def __getitem__(self, audio_ind):
        audio_file = os.path.join(self.folder, self.wav_files[audio_ind])
        class_id = self.target[self.wav_files[audio_ind]]
        
        (sig, sr) = librosa.load(audio_file)

        aud = (sig, sr)
        '''
        if (sr == self.sr):
            resig = sig
        else:
            resig = librosa.resample(sig, sr, self.sr)'''

        reaud = (sig, self.sr)
        resig = sig
        sig_len = resig.shape[0]
        max_len = self.sr//1000 * self.duration
        if len(resig.shape) == 2:
            resig = np.mean(resig, axis = 1)

        if (sig_len > max_len):
            # Truncate the signal to the given length
            start = np.random.randint(0, sig_len-max_len)

            final_sig = resig[start:start+max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = np.zeros((pad_begin_len))
            pad_end = np.zeros((pad_end_len))

            final_sig = np.float32(np.concatenate((pad_begin, resig, pad_end), 0))
            final_aud = (final_sig, self.sr)

        return final_sig, class_id

class WAV2VECGRUSentiment(nn.Module):
    def __init__(self,
                 wav2vec,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.wav2vec = wav2vec
        
        embedding_dim = wav2vec.config.to_dict()['hidden_size']
        
        '''self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)'''
        
        self.out = nn.Linear(128, output_dim)

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, aud):
        aud = aud.squeeze(0)
        hidden_all = list(self.wav2vec(aud).hidden_states)
        embedded = sum(hidden_all)

        embedded = embedded.permute(0, 2, 1)
        embedded = self.relu(self.conv1(embedded))
        embedded = self.relu(self.conv2(embedded))
        hidden = torch.mean(embedded, -1).squeeze(-1)
        #l_1, l_2, l_3, l_4 = outputs[-1], outputs[-2], outputs[-3], outputs[-4]
        #embedded = torch.cat((l_1, l_2, l_3, l_4), axis=2)
        '''_, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])'''
        output = self.out(hidden)
        return output, hidden

def compute_accuracy(output, labels):
    #Function for calculating accuracy
    pred = torch.argmax(output, dim = 1)
    #print(labels)
    correct_pred = (pred == labels).float()
    tot_correct = correct_pred.sum()

    return tot_correct

def compute_loss(output, labels):
    #Function for calculating loss
    weights = [17.57, 2.7, 1.0, 4.25, 6.9, 17.38, 3.91]
    class_weights = torch.FloatTensor(weights).cuda()
    #loss = nn.CrossEntropyLoss(weight=class_weights)(output, labels.squeeze(-1).long())
    ce_loss = nn.CrossEntropyLoss(reduction='none')(output, labels.squeeze(-1).long())
    pt = torch.exp(-ce_loss)
    loss = ((1-pt)**2 * ce_loss).mean()
    return loss

def create_dataset(mode):
    if mode == 'train':
        folder = "train_wavs"
        f = open("train_dict.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "val_wavs"
        f = open("val_dict.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "test_wavs"
        f = open("test_dict.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=32,
                    pin_memory=True,
                    shuffle=True,
                    drop_last=False)
    return loader

def train():
    '''Creates the dataset and the LEAF and CNN model. Trains the LEAF and CNN jointly
    for a total of 100 epochs. The best LEAF and CNN model based on validation loss
    is saved for the next components in the model.
    '''
    train_loader = create_dataset("train")
    val_loader = create_dataset("val")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
    wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h", output_hidden_states=True)

    model = WAV2VECGRUSentiment(wav2vec, 100, 7, 2, True, 0.5)
    model.to(device)
    for name, param in model.named_parameters():
        if 'wav2vec' in name:
            param.requires_grad = False
    base_lr = 1e-4
    optimizer = Adam([{'params':model.parameters(), 'lr':base_lr}])

    final_val_loss = 99999

    for e in range(100):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        train_size = 0
        val_size = 0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(train_loader):
            train_size += data[0].shape[0]
            model.zero_grad()
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = processor(inputs, sampling_rate=16000, return_tensors="pt")
            final_out, _ = model(inputs['input_values'].to(device))
            #correct = compute_accuracy(final_out, labels)
            loss = compute_loss(final_out, labels)
            tot_loss += loss.item()
            #tot_correct += correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(final_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
            print(pred, labels)
            print(f1_score(gt_tr, pred_tr, average='weighted'))
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_size += data[0].shape[0]
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = processor(inputs, sampling_rate=16000, return_tensors="pt")
                val_out, _ = model(inputs['input_values'].to(device))
                #correct = compute_accuracy(val_out, labels)
                loss = compute_loss(val_out, labels)
                val_loss += loss.item()
                #val_correct += correct.item()
                pred = torch.argmax(val_out, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        if val_loss < final_val_loss:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),},
                        'best_model_wavgruorig.tar')
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_loader)
        train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
        #train_acc = tot_correct/train_size
        val_loss_log = val_loss/len(val_loader)
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        #val_acc_log = val_correct/val_size
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")

if __name__ == "__main__":
    '''processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    y, sr = librosa.load('train_wavs/dia0_utt0.wav', None)
    print(y.shape)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    print(inputs['input_values'].shape)
    with torch.no_grad():
        outputs = model(inputs['input_values'])'''
    train()