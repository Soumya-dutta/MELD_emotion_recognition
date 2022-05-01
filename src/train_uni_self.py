from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM
from config import USE_TEXT, USE_AUDIO, LR, EPOCHS, USE_ASR
from config import unimodal_folder, GRU_DIM
import sys
import os
import logging
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)
#device = torch.device("cpu")
class MyDataset(Dataset):
    def __init__(self, aud, text, asr, target, length, mask):

        self.aud = aud
        self.text = text
        self.asr = asr
        self.target = target
        self.length = length
        self.mask = mask
        
    def __getitem__(self, index):
        
        return {'aud': self.aud[index], 'asr':self.asr[index],
                'text': self.text[index], 'target': self.target[index],
                'length':self.length[index], 'speaker_mask':self.mask[index]}
        '''return {'aud': [], 'asr':[],
                'text': self.text[index], 'target': self.target[index],
                'length':self.length[index]}'''
    def __len__(self):
        return len(self.text)

class SelfAttentionModel(nn.Module):
    def __init__(self, hidden_dim, hidden_size=HIDDEN_SIZE, num_atten=NUM_ATTENTION_HEADS):
        super().__init__()
        self.inter_dim = hidden_size//num_atten
        self.num_heads = num_atten
        self.fc_q = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_k = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_v = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        #self.fc_s = nn.Linear(hidden_dim, hidden_dim)
        
        self.multihead_attn = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                    self.num_heads,
                                                    dropout = 0.1,
                                                    bias = True)
        
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps = 1e-6)
        #self.layer_norm_s = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.fc = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        
    
    def forward(self, mod, mask, req, speaker = False):
        mod = mod.double()
        mask = mask.float()
        mask.to(device)
        #print(mask.shape)
        q = self.fc_q(mod)
        k = self.fc_k(mod)
        v = self.fc_v(mod)
        main_diag = [1 for i in range(mask.shape[1])]
        main_diag = np.diag(main_diag)
        mask_new = main_diag
        for reach in range(1, req):
            off_diag = [1 for i in range(mask.shape[1]-reach)]
            up_diag = np.diag(off_diag, reach)
            low_diag = np.diag(off_diag, -1*reach)
            mask_new = mask_new + up_diag + low_diag 
        mask_new = mask_new.astype(bool)
        mask_new = torch.tensor(mask_new)
        mask_new = torch.logical_not(mask_new)
        mask_new = mask_new.to(device)
        if speaker == True:
            self_atten = self.multihead_attn(q, k, v,  need_weights = False, attn_mask = mask)[0]
        else:
            self_atten = self.multihead_attn(q, k, v,  need_weights = False, attn_mask = mask_new)[0]
        mod_q = self.dropout(self.fc(self_atten))
        
        mod_q += mod
        mod_q = self.layer_norm(mod_q)

        #mod_q_s = self.dropout(self.fc_s(mod_q))
        #mod_q_s += mod_q
        #mod_q_s = self.layer_norm_s(mod_q_s)
        
        return mod_q

class GRUModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, layers, bidirectional_flag):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.hidden_dim = hidden_dim
        self.units = nn.ModuleList()
        self.speaker_units = nn.ModuleList()
        self.listener_units = nn.ModuleList()
        self.local_attn = nn.ModuleList()
        self.rnn_1 = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        self.rnn_2 = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        self.rnn_3 = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        self.rnn_4 = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        self.heads = 3
        for ind in range(2):
            self.units.append(SelfAttentionModel(hidden_dim*2, 360, self.heads))
        for ind in range(2):
            self.speaker_units.append(SelfAttentionModel(hidden_dim*2, 360, self.heads))
        for ind in range(2):
            self.listener_units.append(SelfAttentionModel(hidden_dim*2, 360, self.heads))
        for ind in range(2):
            self.local_attn.append(SelfAttentionModel(hidden_dim*2, 360, self.heads))
        self.bidirectional_used = bidirectional_flag
        # self.gate_w_x = nn.Linear(4*hidden_dim, 1)
        # self.gate_w_o = nn.Linear(4*hidden_dim, 1)
        self.fc_1 = nn.Linear(input_dim, hidden_dim*8)#+ input_dim)
        #self.fc_2 = nn.Linear(hidden_dim*8, 200)
        self.conv1 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=1)
        self.fc = nn.Linear(hidden_dim*8, output_dim)
        #self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, length, speaker_mask):
        mask = torch.logical_not(length)
        mask.to(device)
        speaker_mask.to(device)
        output_gru_1, _ = self.rnn_1(x)
        output_gru_1 = output_gru_1.permute(1, 0, 2)
        output_gru_2, _ = self.rnn_2(x)
        output_gru_2 = output_gru_2.permute(1, 0, 2)
        output_gru_3, _ = self.rnn_3(x)
        output_gru_3 = output_gru_3.permute(1, 0, 2)
        output_gru_4, _ = self.rnn_4(x)
        output_gru_4 = output_gru_4.permute(1, 0, 2)
        #output = x.permute(1, 0, 2)
        speaker_mask = speaker_mask.repeat(self.heads, 1, 1)
        permuted_x = x.permute(1, 0, 2)
        for attn_model in self.units:
            permuted_x = output_gru_1
            output_con = attn_model(permuted_x, mask, 33)
            output_con = output_con.permute(1, 0, 2)
            output_con = output_con.permute(0, 2, 1)
            output_con = self.relu(self.conv1(output_con))
            output_con = output_con.permute(0, 2, 1)
        for attn_model in self.local_attn:
            permuted_x = output_gru_2
            output_local_con = attn_model(permuted_x, mask, 2)
            output_local_con = output_local_con.permute(1, 0, 2)
            output_local_con = output_local_con.permute(0, 2, 1)
            output_local_con = self.relu(self.conv2(output_local_con))
            output_local_con = output_local_con.permute(0, 2, 1)
        for attn_model in self.speaker_units:
            permuted_x = output_gru_3
            output_speaker = attn_model(permuted_x, speaker_mask, 33,True)
            output_speaker = output_speaker.permute(1, 0, 2)
            output_speaker = output_speaker.permute(0, 2, 1)
            output_speaker = self.relu(self.conv3(output_speaker))
            output_speaker = output_speaker.permute(0, 2, 1)
        for attn_model in self.listener_units:
            permuted_x = output_gru_4
            listener_mask = torch.logical_not(speaker_mask)
            listener_mask.to(device)
            output_listener = attn_model(permuted_x, listener_mask, 33, True)
            output_listener = output_listener.permute(1, 0, 2)
            output_listener = output_listener.permute(0, 2, 1)
            output_listener = self.relu(self.conv4(output_listener))
            output_listener = output_listener.permute(0, 2, 1)

        #print(output_con-output_speaker)
        output = torch.cat((output_con, output_speaker, output_listener, output_local_con), -1)
        #output = output_con + output_speaker
        #output = torch.nan_to_num(output, nan=1e-6)
        #print(output)
        x_1 = self.fc_1(x)
        #gate_val = self.gate_w_x(x_1) + self.gate_w_o(output)
        #gate_val = self.sig(gate_val)
        output += x_1 # +(1-gate_val)*output
        #output = self.dropout(output)
        #output = self.dropout(self.fc_2(output))
        #out = self.fc(output)
        out = self.fc(output)
        return output, out


def compute_accuracy(output, req_dic):
    batch_correct = 0.0
    batch_total = 0.0
    tot_pred = []
    tot_labels = []
    for i in range(output.shape[0]):
        req_len = torch.sum(req_dic['length'][i]).int()
        out_required = output[i][:req_len, :]
        target_required = req_dic['target'][i][:req_len].int()
        pred = torch.argmax(out_required, dim = -1)
        pred = pred.detach().cpu().numpy()
        pred = list(pred)
        labels = target_required.detach().cpu().numpy()
        labels = list(labels)
        tot_pred.extend(pred)
        tot_labels.extend(labels)
        #correct_pred = (pred == target_required).float()
        #tot_correct = correct_pred.sum()
        #batch_correct += tot_correct
        #batch_total += req_len
    return tot_pred, tot_labels

def compute_loss(output, train_dic):
    batch_loss = 0.0
    weights = [17.57, 2.7, 1.0, 4.25, 6.9, 17.38, 3.91]
    class_weights = torch.DoubleTensor(weights).to(device)
    #print(output[0, :, :], train_dic['target'])
    for i in range(output.shape[0]):
        req_len = torch.sum(train_dic['length'][i]).int()
        
        '''loss = nn.CrossEntropyLoss(ignore_index = 7, weight=class_weights)(output[i][:req_len, :],
                                                     train_dic['target'][i][:req_len].long().to(device))'''
        ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=7)(output[i][:req_len, :], train_dic['target'][i][:req_len].long().to(device))
        #print(ce_loss)
        #ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=7)(output[i], train_dic['target'][i].long().to(device))
        pt = torch.exp(-ce_loss)
        loss = ((1-pt)**2 * ce_loss).mean()
        #loss = loss*33/req_len
        batch_loss += loss
    return batch_loss/output.shape[0]

def count_parameters(model):
    print("NUM PARAMETERS", sum(p.numel() for p in model.parameters() if p.requires_grad))

def train(train_data, val_data, max_length):
    base_lr = LR
    n_gpu = torch.cuda.device_count()
    
    final_val_loss = 999999
    train_loader = train_data
    val_loader = val_data
    
    if USE_TEXT:
        model = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    if USE_AUDIO:
        model = GRUModel(AUDIO_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    
    model.to(device)
    count_parameters(model)
    optimizer = Adam(model.parameters(), lr=base_lr)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for e in range(EPOCHS):
        print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))
        tot_loss, tot_acc = 0.0, 0.0
        model.train()
        pred_tr = []
        gt_tr = []
        for ind, train_dic in enumerate(train_loader):
            model.zero_grad()
            if USE_AUDIO:
                inp = train_dic['aud'].permute(0, 2, 1).double()
            if USE_TEXT:
                if USE_ASR:
                    inp = train_dic['asr'].permute(0, 2, 1).double()
                else:
                    inp = train_dic['text'].permute(0, 2, 1).double()
            length = train_dic['length'].to(device)
            speaker_mask = train_dic['speaker_mask'].to(device)
            _, out = model(inp.to(device), length, speaker_mask)
            train_dic['target'][train_dic['target'] == -1] = 7
            pred, gt = compute_accuracy(out.cpu(), train_dic)
            pred_tr.extend(pred)
            gt_tr.extend(gt)
            loss = compute_loss(out.to(device), train_dic)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            pred_val = []
            gt_val = []
            for ind, val_dic in enumerate(val_loader):
                if USE_AUDIO:
                    inp = val_dic['aud'].permute(0, 2, 1).double()
                if USE_TEXT:
                    if USE_ASR:
                        inp = val_dic['asr'].permute(0, 2, 1).double()
                    else:
                        inp = val_dic['text'].permute(0, 2, 1).double()
                length = val_dic['length'].to(device)
                speaker_mask = val_dic['speaker_mask'].to(device)
                _, val_out = model(inp.to(device), length.to(device), speaker_mask)

                val_dic['target'][val_dic['target'] == -1] = 7
                pred, gt = compute_accuracy(val_out.cpu(), val_dic)
                val_loss += compute_loss(val_out.to(device), val_dic).item()
                pred_val.extend(pred)
                gt_val.extend(gt)
            if val_loss < final_val_loss:
                if USE_AUDIO:
                    if USE_ASR:
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),},
                                    'asr_aud_self_best_model.tar')
                    else:
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),},
                                    'aud_self_best_model.tar')
                if USE_TEXT:
                    if USE_ASR:
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),},
                                    'asr_text_self_best_model.tar')
                    else:
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),},
                                    'text_self_best_model.tar')
                final_val_loss = val_loss
            e_log = e + 1
            train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
            val_f1 = f1_score(gt_val, pred_val, average='weighted')
            train_loss = tot_loss/len(train_loader)
            val_loss_log = val_loss/len(val_loader)
            logger.info(f"Epoch {e_log}, \
                        Training Loss {train_loss},\
                        Training Accuracy {train_f1}")
            logger.info(f"Epoch {e_log}, \
                        Validation Loss {val_loss_log},\
                        Validation Accuracy {val_f1}")
