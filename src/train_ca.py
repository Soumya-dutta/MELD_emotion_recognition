from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM
from config import USE_TEXT, USE_AUDIO, LR, EPOCHS
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
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from src.train_uni_self import GRUModel, compute_accuracy, compute_loss, MyDataset
from sklearn.metrics import f1_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.inter_dim = HIDDEN_SIZE//NUM_ATTENTION_HEADS
        self.num_heads = NUM_ATTENTION_HEADS
        self.fc_audq = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audk = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audv = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_textq = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_textk = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_textv = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audq_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audk_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audv_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_textq_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_textk_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_textv_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.multihead_attn_text = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.0,
                                                         bias = True)
        self.multihead_attn_aud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.0,
                                                         bias = True)
        self.multihead_attn_selfaud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.0,
                                                         bias = True)
        self.multihead_attn_selftext = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.0,
                                                         bias = True)
        self.layer_norm_t = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.layer_norm_a = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.layer_norm_t_s = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.layer_norm_a_s = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.fc_text_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_audio_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_text = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        self.fc_audio = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        self.fc_text_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        self.fc_aud_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, text, audio, mask):
        text = text.float()
        audio = audio.float()
        mask.to(device)
        text_q = self.fc_textq(text)
        text_k = self.fc_textk(text)
        text_v = self.fc_textv(text)
        audio_q = self.fc_audq(audio)
        audio_k = self.fc_audk(audio)
        audio_v = self.fc_audv(audio)
        text_cross = self.multihead_attn_text(text_q, audio_k, audio_v, key_padding_mask = mask, need_weights = False)[0]
        audio_cross = self.multihead_attn_aud(audio_q, text_k, text_v, key_padding_mask = mask, need_weights = False)[0]
        text_q = self.dropout(self.fc_text(text_cross))
        audio_q = self.dropout(self.fc_audio(audio_cross))
        text_q += text
        audio_q += audio
        text_q = self.layer_norm_t(text_q)
        audio_q = self.layer_norm_a(audio_q)
        text_q_s = self.fc_textq_s(text_q)
        text_k_s = self.fc_textk_s(text_q)
        text_v_s = self.fc_textv_s(text_q)
        aud_q_s = self.fc_audq_s(audio_q)
        aud_k_s = self.fc_audk_s(audio_q)
        aud_v_s = self.fc_audv_s(audio_q)
        text_self = self.multihead_attn_selftext(text_q_s, text_k_s, text_v_s, key_padding_mask = mask, need_weights = False)[0]
        aud_self = self.multihead_attn_selfaud(aud_q_s, aud_k_s, aud_v_s, key_padding_mask = mask, need_weights = False)[0]
        text_q_s = self.dropout(self.fc_text_s(text_self))
        aud_q_s = self.dropout(self.fc_aud_s(aud_self))
        text_q_s += text_q
        aud_q_s += audio_q
        text_q_fin = self.layer_norm_t_s(text_q_s)
        aud_q_fin = self.layer_norm_a_s(aud_q_s)
        text_q_1 = self.dropout(self.fc_text_1(text_q))
        audio_q_1 = self.dropout(self.fc_audio_1(audio_q))
        text_q_1 += text_q
        audio_q_1 += audio_q
        return text_q_1, audio_q_1

class FusionModule(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.fc = nn.Linear(hidden_dim*2, 7)
        self.units = nn.ModuleList()
        self.bn = nn.LayerNorm(hidden_dim*2, eps = 1e-6)
        #self.recon_text = nn.Linear(hidden_dim, hidden_dim)
        #self.recon_audio = nn.Linear(hidden_dim, hidden_dim)
        for ind in range(n_layers):
            self.units.append(CrossAttentionModel(hidden_dim))
    
    def forward(self, text, audio, length):
        mask = torch.logical_not(length)
        mask.to(device)
        text = text.permute(1,0,2)
        audio = audio.permute(1,0,2)
        for model_ca in self.units:
            text, audio = model_ca(text, audio, mask)
        #text_recon = self.recon_text(audio)
        #audio_recon = self.recon_audio(text)
        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)
        concat = torch.cat((text, audio), dim = -1)
        concat = self.bn(concat)
        output = self.fc(concat)
        return output#, text_recon, audio_recon

def count_parameters(model):
    print("NUM PARAMETERS", sum(p.numel() for p in model.parameters() if p.requires_grad))

def train(train_data, val_data, max_length):
    base_lr = LR
    n_gpu = torch.cuda.device_count()
    
    final_val_loss = 999999
    train_loader = train_data
    val_loader = val_data
    model = FusionModule(GRU_DIM*4, NUM_HIDDEN_LAYERS)
    model.to(device)
    count_parameters(model)
    if USE_TEXT:
        text_model = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 1, True).double()
        text_model.to(device)
        checkpoint_text = torch.load('text_self_best_model.tar')
        text_model.load_state_dict(checkpoint_text['model_state_dict'])
        text_model.eval()
    if USE_AUDIO:
        aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 7, 0.5, 2, True).double()
        aud_model.to(device)
        checkpoint_aud = torch.load('aud_self_best_model.tar')
        aud_model.load_state_dict(checkpoint_aud['model_state_dict'])
        aud_model.eval()

    optimizer = Adam([{'params':model.parameters()}], lr=base_lr)
    scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    for e in range(EPOCHS):
        print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))
        tot_loss = 0.0
        model.train()
        #aud_model.train()
        #text_model.train()
        pred_tr = []
        gt_tr = []
        for ind, train_dic in enumerate(train_loader):
            model.zero_grad()
            length = train_dic['length']
            speaker_mask = train_dic['speaker_mask'].to(device)
            if USE_AUDIO:
                inp = train_dic['aud'].permute(0, 2, 1).double()
                train_dic['aud'], _ = aud_model(inp.to(device), length.to(device), speaker_mask)
            if USE_TEXT:
                inp = train_dic['text'].permute(0, 2, 1).double()
                train_dic['text'], _ = text_model.forward(inp.to(device), length.to(device), speaker_mask)

            out = model(train_dic['text'], train_dic['aud'], train_dic['length'].to(device))
            train_dic['target'][train_dic['target'] == -1] = 7
            pred, gt = compute_accuracy(out.cpu(), train_dic)
            loss = compute_loss(out.to(device), train_dic)
            tot_loss += loss.item()
            pred_tr.extend(pred)
            gt_tr.extend(gt)
            loss.backward()
            optimizer.step()

        model.eval()
        scheduler.step()
        with torch.no_grad():
            val_loss = 0.0
            pred_val = []
            gt_val = []
            for ind, val_dic in enumerate(val_loader):
                length = val_dic['length']
                speaker_mask = val_dic['speaker_mask'].to(device)
                if USE_AUDIO:
                    inp = val_dic['aud'].permute(0, 2, 1).double()
                    val_dic['aud'], _ = aud_model.forward(inp.to(device), length.to(device), speaker_mask)
                if USE_TEXT:
                    inp = val_dic['text'].permute(0, 2, 1).double()
                    val_dic['text'], _ = text_model.forward(inp.to(device), length.to(device), speaker_mask)

                val_out = model(val_dic['text'], val_dic['aud'], val_dic['length'].to(device))
                val_dic['target'][val_dic['target'] == -1] = 7
                pred, gt = compute_accuracy(val_out.cpu(), val_dic)
                pred_val.extend(pred)
                gt_val.extend(gt)
                val_loss += compute_loss(val_out.to(device), val_dic).item()
            if val_loss < final_val_loss:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                            'cross_best_model.tar')
                final_val_loss = val_loss
            e_log = e + 1
            train_loss = tot_loss/len(train_loader)
            train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
            val_f1 = f1_score(gt_val, pred_val, average='weighted')
            val_loss_log = val_loss/len(val_loader)
            logger.info(f"Epoch {e_log}, \
                        Training Loss {train_loss},\
                        Training Accuracy {train_f1}")
            logger.info(f"Epoch {e_log}, \
                        Validation Loss {val_loss_log},\
                        Validation Accuracy {val_f1}")
