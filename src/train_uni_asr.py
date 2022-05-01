from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM
from config import USE_TEXT, USE_AUDIO, LR, EPOCHS
from config import unimodal_folder, GRU_DIM
from config import test_npy_audio_path, asr_test_npy_text_path,test_npy_label_path, test_npy_text_path
from config import length_test
from src.train_uni_self import GRUModel, MyDataset
from src.train_uni_self import compute_accuracy, compute_loss
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


class CrossAttentionModel_s(nn.Module):
    def __init__(self, hidden_dim, hidden_size=HIDDEN_SIZE, num_atten=NUM_ATTENTION_HEADS):
        super().__init__()
        self.inter_dim = hidden_size//num_atten
        self.num_heads = num_atten
        self.fc_q = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_k = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_v = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)  
        '''self.fc_q_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_k_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_v_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)'''        
        self.multihead_attn = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                    self.num_heads,
                                                    dropout = 0.5,
                                                    bias = True)
        '''self.multihead_attn_s = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                    self.num_heads,
                                                    dropout = 0.0,
                                                    bias = True,
                                                    batch_first=True)'''                     
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps = 1e-6)
        #self.layer_norm_s = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.fc = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        #self.fc_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
    
    def forward(self, aud, text, mask):
        aud = aud.double()
        text = text.double()
        #mask = mask.double()
        mask.to(device)
        q = self.fc_q(aud)
        k = self.fc_k(text)
        v = self.fc_v(text)
        
        self_atten = self.multihead_attn(q, k, v, key_padding_mask = mask, need_weights = False)[0]
        mod_q = self.dropout(self.fc(self_atten))
        
        mod_q += aud
        
        mod_q = self.layer_norm(mod_q)
        '''
        q_s = self.fc_q_s(mod_q)
        k_s = self.fc_k_s(mod_q)
        v_s = self.fc_v_s(mod_q)

        self_atten_s = self.multihead_attn(q_s, k_s, v_s, key_padding_mask = mask, need_weights = False)[0]
        mod_q_s = self.dropout(self.fc_s(self_atten_s))
        
        mod_q_s += mod_q
        
        mod_q_s = self.layer_norm_s(mod_q_s)'''

        return mod_q

class FusionModule(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.fc = nn.Linear(400, 7)
        self.units = nn.ModuleList()
        self.fc_aud = nn.Linear(hidden_dim, 400)
        self.fc_text = nn.Linear(hidden_dim, 400)
        self.bn = nn.LayerNorm(400, eps = 1e-6)
        #self.recon_text = nn.Linear(hidden_dim, hidden_dim)
        #self.recon_audio = nn.Linear(hidden_dim, hidden_dim)
        # self.conv = nn.Conv1d(in_channels=400, out_channels=400, kernel_size=1)
        # self.conv_1 = nn.Conv1d(in_channels=400, out_channels=400, kernel_size=1)
        self.relu = nn.ReLU()
        for ind in range(n_layers):
            self.units.append(CrossAttentionModel_s(400, 60, 3))
    
    def forward(self, audio, text, length):
        mask = torch.logical_not(length)
        mask.to(device)
        audio = self.fc_aud(audio)
        text = self.fc_text(text)
        audio = audio.permute(1, 0, 2)
        text = text.permute(1, 0, 2)

        for model_ca in self.units:
            audio = model_ca(audio, text, mask)
        audio = audio.permute(1, 0, 2)
        # audio = audio.permute(0, 2, 1)
        # audio = self.relu(self.conv(audio))
        # audio = audio.permute(0, 2, 1)
        #text_recon = self.recon_text(audio)
        #audio_recon = self.recon_audio(text)
        concat = self.bn(audio)
        output = self.fc(concat)
        return concat, output#, text_recon, audio_recon
'''
class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim_a, hidden_dim_t):
        super().__init__()
        self.inter_dim = HIDDEN_SIZE//NUM_ATTENTION_HEADS
        self.num_heads = NUM_ATTENTION_HEADS
        # self.fc_audq = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        # self.fc_audk = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        # self.fc_audv = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        # self.fc_textq = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        # self.fc_textk = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        # self.fc_textv = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        self.conv_audq = nn.Conv1d(in_channels=hidden_dim_a, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        self.conv_audk = nn.Conv1d(in_channels=hidden_dim_a, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        self.conv_audv = nn.Conv1d(in_channels=hidden_dim_a, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        self.conv_textq = nn.Conv1d(in_channels=hidden_dim_t, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        self.conv_textk = nn.Conv1d(in_channels=hidden_dim_t, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        self.conv_textv = nn.Conv1d(in_channels=hidden_dim_t, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        # self.fc_audq_s = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        # self.fc_audk_s = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        # self.fc_audv_s = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        # self.fc_textq_s = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        # self.fc_textk_s = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        # self.fc_textv_s = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        self.multihead_attn_text = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.1,
                                                         bias = True)
        self.multihead_attn_aud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.1,
                                                         bias = True)
        
        self.multihead_attn_selfaud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.1,
                                                         bias = True)
        self.multihead_attn_selftext = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.1,
                                                         bias = True)                                                                                                    
        self.dropout = nn.Dropout(0.5)
        self.layer_norm_t = nn.LayerNorm(hidden_dim_t, eps = 1e-6)
        self.layer_norm_a = nn.LayerNorm(hidden_dim_a, eps = 1e-6)
        # self.layer_norm_t_s = nn.LayerNorm(hidden_dim_t, eps = 1e-6)
        # self.layer_norm_a_s = nn.LayerNorm(hidden_dim_a, eps = 1e-6)
        #self.layer_norm_t_f = nn.LayerNorm(hidden_dim_t, eps = 1e-6)
        #self.layer_norm_a_f = nn.LayerNorm(hidden_dim_a, eps = 1e-6)
        #self.fc_text_1 = nn.Linear(hidden_dim_t, hidden_dim_t)
        #self.fc_audio_1 = nn.Linear(hidden_dim_a, hidden_dim_a)
        self.fc_text = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_t)
        self.fc_audio = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_a)
        self.relu = nn.ReLU()
        # self.fc_text_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_t)
        # self.fc_aud_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_a)
    
    def forward(self, text, audio, mask):
        text = text.float()
        audio = audio.float()
        mask.to(device)
        
        text_q = text.permute(0, 2, 1)
        text_q = self.relu(self.conv_textq(text_q))
        text_q = text_q.permute(0, 2, 1)
        text_k = text.permute(0, 2, 1)
        text_k = self.relu(self.conv_textk(text_k))
        text_k = text_k.permute(0, 2, 1)
        text_v = text.permute(0, 2, 1)
        text_v = self.relu(self.conv_textv(text_v))
        text_v = text_v.permute(0, 2, 1)

        audio_q = audio.permute(0, 2, 1)
        audio_q = self.relu(self.conv_audq(audio_q))
        audio_q = audio_q.permute(0, 2, 1)
        audio_k = audio.permute(0, 2, 1)
        audio_k = self.relu(self.conv_audk(audio_k))
        audio_k = audio_k.permute(0, 2, 1)
        audio_v = audio.permute(0, 2, 1)
        audio_v = self.relu(self.conv_audv(audio_v))
        audio_v = audio_v.permute(0, 2, 1)

        # text_q = self.fc_textq(text)
        # text_k = self.fc_textk(text)
        # text_v = self.fc_textv(text)
        # audio_q = self.fc_audq(audio)
        # audio_k = self.fc_audk(audio)
        # audio_v = self.fc_audv(audio)
        text_cross = self.multihead_attn_text(text_q, audio_k, audio_v, key_padding_mask = mask, need_weights = False)[0]
        audio_cross = self.multihead_attn_aud(audio_q, text_k, text_v, key_padding_mask = mask, need_weights = False)[0]
        text_q = self.dropout(self.fc_text(text_cross))
        audio_q = self.dropout(self.fc_audio(audio_cross))
        text_q += text
        audio_q += audio
        text_q = self.layer_norm_t(text_q)
        audio_q = self.layer_norm_a(audio_q)
        # text_q_s = self.fc_textq_s(text_q)
        # text_k_s = self.fc_textk_s(text_q)
        # text_v_s = self.fc_textv_s(text_q)
        # aud_q_s = self.fc_audq_s(audio_q)
        # aud_k_s = self.fc_audk_s(audio_q)
        # aud_v_s = self.fc_audv_s(audio_q)
        # text_self = self.multihead_attn_selftext(text_q_s, text_k_s, text_v_s, key_padding_mask = mask, need_weights = False)[0]
        # aud_self = self.multihead_attn_selfaud(aud_q_s, aud_k_s, aud_v_s, key_padding_mask = mask, need_weights = False)[0]
        # text_q_s = self.dropout(self.fc_text_s(text_self))
        # aud_q_s = self.dropout(self.fc_aud_s(aud_self))
        # text_q_s += text_q
        # aud_q_s += audio_q
        # text_q_fin = self.layer_norm_t_s(text_q_s)
        # aud_q_fin = self.layer_norm_a_s(aud_q_s)
        #text_q_1 = self.dropout(self.fc_text_1(text_q_fin))
        #audio_q_1 = self.dropout(self.fc_audio_1(aud_q_fin))

        #text_q_1 += text_q_fin
        #audio_q_1 += aud_q_fin
        #text_q_1 = self.layer_norm_t_f(text_q_1)
        #audio_q_1 = self.layer_norm_a_f(audio_q_1)
        return text_q, audio_q
        #return text_q_1, audio_q_1

class FusionModuleAll(nn.Module):
    def __init__(self, hidden_dim_a, hidden_dim_t, n_layers):
        super().__init__()
        self.hid = 400
        self.fc = nn.Linear(self.hid*2, 7)
        self.fc_text = nn.Linear(hidden_dim_t, self.hid)
        self.fc_aud = nn.Linear(hidden_dim_a, self.hid)
        # self.fc_text_1 = nn.Linear(hidden_dim_t, self.hid)
        # self.fc_aud_1 = nn.Linear(hidden_dim_a, self.hid)
        self.conv1 = nn.Conv1d(in_channels=self.hid, out_channels=self.hid, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.hid, out_channels=self.hid, kernel_size=1)
        self.units = nn.ModuleList()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.LayerNorm(self.hid*2, eps = 1e-6)
        self.relu = nn.ReLU()
        #self.recon_text = nn.Linear(hidden_dim, hidden_dim)
        #self.recon_audio = nn.Linear(hidden_dim, hidden_dim)
        for ind in range(n_layers):
            self.units.append(CrossAttentionModel(self.hid, self.hid))
    
    def forward(self, text_orig, audio_orig, length):
        mask = torch.logical_not(length)
        mask.to(device)
        
        text = text_orig.permute(1, 0, 2).float()
        audio = audio_orig.permute(1, 0, 2).float()
        text = self.fc_text(text)
        audio = self.fc_aud(audio)
        for model_ca in self.units:
            text, audio = model_ca(text, audio, mask)
        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)
        text = text.permute(0, 2, 1)
        text = self.relu(self.conv1(text))
        text = text.permute(0, 2, 1)
        audio = audio.permute(0, 2, 1)
        audio = self.relu(self.conv2(audio))
        audio = audio.permute(0, 2, 1)
        # text_orig = self.fc_text(text_orig.float())
        # audio_orig = self.fc_aud(audio_orig.float())
        # text += text_orig
        # audio += audio_orig
        #text_recon = self.recon_text(audio)
        #audio_recon = self.recon_audio(text)
        concat = self.bn(torch.cat((text, audio), dim = -1))
        output = self.fc(concat)
        return concat, output#, text_recon, audio_recon'''

class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim_a, hidden_dim_t):
        super().__init__()
        self.inter_dim = HIDDEN_SIZE//NUM_ATTENTION_HEADS
        self.num_heads = NUM_ATTENTION_HEADS
        self.fc_audq = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        self.fc_audk = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        self.fc_audv = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        self.fc_textq = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        self.fc_textk = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        self.fc_textv = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        # self.conv_audq = nn.Conv1d(in_channels=hidden_dim_a, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        # self.conv_audk = nn.Conv1d(in_channels=hidden_dim_a, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        # self.conv_audv = nn.Conv1d(in_channels=hidden_dim_a, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        # self.conv_textq = nn.Conv1d(in_channels=hidden_dim_t, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        # self.conv_textk = nn.Conv1d(in_channels=hidden_dim_t, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        # self.conv_textv = nn.Conv1d(in_channels=hidden_dim_t, out_channels=self.inter_dim*self.num_heads, kernel_size=1)
        self.fc_audq_s = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        self.fc_audk_s = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        self.fc_audv_s = nn.Linear(hidden_dim_a, self.inter_dim*self.num_heads)
        self.fc_textq_s = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        self.fc_textk_s = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        self.fc_textv_s = nn.Linear(hidden_dim_t, self.inter_dim*self.num_heads)
        self.multihead_attn_text = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.0,
                                                         bias = False)
        self.multihead_attn_aud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.0,
                                                         bias = False)
        
        self.multihead_attn_selfaud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.0,
                                                         bias = False)
        self.multihead_attn_selftext = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.0,
                                                         bias = False)                                                                                                    
        self.dropout = nn.Dropout(0.5)
        self.layer_norm_t = nn.LayerNorm(hidden_dim_t, eps = 1e-6)
        self.layer_norm_a = nn.LayerNorm(hidden_dim_a, eps = 1e-6)
        self.layer_norm_t_s = nn.LayerNorm(hidden_dim_t, eps = 1e-6)
        self.layer_norm_a_s = nn.LayerNorm(hidden_dim_a, eps = 1e-6)
        #self.layer_norm_t_f = nn.LayerNorm(hidden_dim_t, eps = 1e-6)
        #self.layer_norm_a_f = nn.LayerNorm(hidden_dim_a, eps = 1e-6)
        #self.fc_text_1 = nn.Linear(hidden_dim_t, hidden_dim_t)
        #self.fc_audio_1 = nn.Linear(hidden_dim_a, hidden_dim_a)
        self.fc_text = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_t)
        self.fc_audio = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_a)
        self.relu = nn.ReLU()
        self.fc_text_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_t)
        self.fc_aud_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_a)
    
    def forward(self, text, audio, mask):
        text = text.float()
        audio = audio.float()
        mask.to(device)
        
        # text_q = text.permute(0, 2, 1)
        # text_q = self.relu(self.conv_textq(text_q))
        # text_q = text_q.permute(0, 2, 1)
        # text_k = text.permute(0, 2, 1)
        # text_k = self.relu(self.conv_textk(text_k))
        # text_k = text_k.permute(0, 2, 1)
        # text_v = text.permute(0, 2, 1)
        # text_v = self.relu(self.conv_textv(text_v))
        # text_v = text_v.permute(0, 2, 1)

        # audio_q = audio.permute(0, 2, 1)
        # audio_q = self.relu(self.conv_audq(audio_q))
        # audio_q = audio_q.permute(0, 2, 1)
        # audio_k = audio.permute(0, 2, 1)
        # audio_k = self.relu(self.conv_audk(audio_k))
        # audio_k = audio_k.permute(0, 2, 1)
        # audio_v = audio.permute(0, 2, 1)
        # audio_v = self.relu(self.conv_audv(audio_v))
        # audio_v = audio_v.permute(0, 2, 1)

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
        #text_q_1 = self.dropout(self.fc_text_1(text_q_fin))
        #audio_q_1 = self.dropout(self.fc_audio_1(aud_q_fin))

        #text_q_1 += text_q_fin
        #audio_q_1 += aud_q_fin
        #text_q_1 = self.layer_norm_t_f(text_q_1)
        #audio_q_1 = self.layer_norm_a_f(audio_q_1)
        #return text_q, audio_q
        return text_q_fin, aud_q_fin

class FusionModuleAll(nn.Module):
    def __init__(self, hidden_dim_a, hidden_dim_t, n_layers):
        super().__init__()
        self.hid = 400
        self.fc = nn.Linear(self.hid*2, 7)
        self.fc_text = nn.Linear(hidden_dim_t, self.hid)
        self.fc_aud = nn.Linear(hidden_dim_a, self.hid)
        # self.fc_text_1 = nn.Linear(hidden_dim_t, self.hid)
        # self.fc_aud_1 = nn.Linear(hidden_dim_a, self.hid)
        # self.conv1 = nn.Conv1d(in_channels=self.hid, out_channels=self.hid, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=self.hid, out_channels=self.hid, kernel_size=1)
        self.units = nn.ModuleList()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.LayerNorm(self.hid*2, eps = 1e-6)
        self.relu = nn.ReLU()
        #self.recon_text = nn.Linear(hidden_dim, hidden_dim)
        #self.recon_audio = nn.Linear(hidden_dim, hidden_dim)
        for ind in range(n_layers):
            self.units.append(CrossAttentionModel(self.hid, self.hid))
    
    def forward(self, text_orig, audio_orig, length):
        mask = torch.logical_not(length)
        mask.to(device)
        
        text = text_orig.permute(1, 0, 2).float()
        audio = audio_orig.permute(1, 0, 2).float()
        text = self.relu(self.fc_text(text))
        audio = self.relu(self.fc_aud(audio))
        for model_ca in self.units:
            text, audio = model_ca(text, audio, mask)
        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)
        # text = text.permute(0, 2, 1)
        # text = self.relu(self.conv1(text))
        # text = text.permute(0, 2, 1)
        # audio = audio.permute(0, 2, 1)
        # audio = self.relu(self.conv2(audio))
        # audio = audio.permute(0, 2, 1)
        # text_orig = self.fc_text(text_orig.float())
        # audio_orig = self.fc_aud(audio_orig.float())
        # text += text_orig
        # audio += audio_orig
        #text_recon = self.recon_text(audio)
        #audio_recon = self.recon_audio(text)
        concat = self.bn(torch.cat((text, audio), dim = -1))
        output = self.fc(concat)
        return concat, output#, text_recon, audio_recon

def test_model(PATH):
    test_aud = np.load(test_npy_audio_path)
    test_text = np.load(test_npy_text_path)
    test_labels = np.load(test_npy_label_path)
    test_lengths = np.load(length_test)
    max_length = test_lengths.shape[1]
    asr = np.load(asr_test_npy_text_path)
    test_dataset = MyDataset(test_aud, test_text, asr, test_labels,
                            test_lengths)
    test_size = test_aud.shape[0]
    indices = list(range(test_size))
    test_sampler = SubsetRandomSampler(indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    test_loader = DataLoader(test_dataset,
                            sampler = test_sampler,
                            batch_size=64,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False,
                            )
    test_results = dict()
    for fold in range(1):
        test_model = FusionModule(GRU_DIM*2, 3).double()
        #test_model = FusionModuleAll(GRU_DIM*2, GRU_DIM*2, 3)
        model_path = PATH.replace("model", "model_"+str(fold))
        checkpoint = torch.load(model_path)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        test_model.to(device)
        test_model.eval()

        aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 4, 0.5, 2, True).double()
        aud_model.to(device)
        checkpoint_aud = torch.load(os.path.join('aud_self_best_model.tar'))
        aud_model.load_state_dict(checkpoint_aud['model_state_dict'])
        aud_model.eval()

        text_model = GRUModel(TEXT_DIM, GRU_DIM, 4, 0.5, 1, True).double()
        text_model.to(device)
        checkpoint_text = torch.load(os.path.join('asr_text_self_best_model.tar'))
        text_model.load_state_dict(checkpoint_text['model_state_dict'])
        text_model.eval()

        for ind, test_dic in enumerate(test_loader):
            length = test_dic['length']

            inp = test_dic['aud'].permute(0, 2, 1).double()
            test_dic['aud'], _ = aud_model.forward(inp.to(device), length.to(device))
            inp = test_dic['asr'].permute(0, 2, 1).double()
            test_dic['asr'], _ = text_model.forward(inp.to(device), length.to(device))

            _, test_out = test_model.forward(test_dic['aud'], test_dic['asr'], test_dic['length'].to(device))
            #_, test_out = test_model.forward(test_dic['asr'], test_dic['aud'], test_dic['length'].to(device))
            test_dic['target'][test_dic['target'] == -1] = 4
            test_results[fold] = test_out
        acc_fold = compute_accuracy(test_out.cpu(), test_dic).item()
        logger.info(f"Accuracy of fold-  {fold}- {acc_fold}")

def count_parameters(model):
    print("NUM PARAMETERS", sum(p.numel() for p in model.parameters() if p.requires_grad))

def train(train_data, val_data, max_length):
    base_lr = LR
    n_gpu = torch.cuda.device_count()
    final_val_loss = 999999
    train_loader = train_data
    val_loader = val_data

    aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    text_model = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    fusion_model = FusionModule(400*4, 2).double()
    #fusion_model = FusionModuleAll(GRU_DIM*2, GRU_DIM*2, NUM_HIDDEN_LAYERS)
    count_parameters(fusion_model)
    fusion_model.to(device)

    text_model.to(device)
    checkpoint_text = torch.load(os.path.join('asr_text_self_best_model.tar'))
    text_model.load_state_dict(checkpoint_text['model_state_dict'])

    aud_model.to(device)
    checkpoint_aud = torch.load(os.path.join('aud_self_best_model.tar'))
    aud_model.load_state_dict(checkpoint_aud['model_state_dict'])

    optimizer = Adam([{'params':fusion_model.parameters()}], lr=base_lr)
    scheduler = MultiStepLR(optimizer, milestones=[50,175], gamma=0.1)

    for e in range(EPOCHS):
        print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))
        tot_loss, tot_acc = 0.0, 0.0
        fusion_model.train()
        #aud_model.train()
        #text_model.train()
        pred_tr = []
        gt_tr = []
        for ind, train_dic in enumerate(train_loader):
            fusion_model.zero_grad()
            length = train_dic['length'].to(device)
            speaker_mask = train_dic['speaker_mask'].to(device)
            inp = train_dic['aud'].permute(0, 2, 1).double()
            train_dic['aud'], _ = aud_model(inp.to(device), length.to(device), speaker_mask)
            inp = train_dic['asr'].permute(0, 2, 1).double()
            train_dic['asr'], _ = text_model(inp.to(device), length.to(device), speaker_mask)
            _, out = fusion_model(train_dic['aud'], train_dic['asr'], train_dic['length'].to(device))
            #_, out = fusion_model(train_dic['asr'], train_dic['aud'], train_dic['length'].to(device))
            train_dic['target'][train_dic['target'] == -1] = 7
            pred, gt = compute_accuracy(out.cpu(), train_dic)
            loss = compute_loss(out.to(device), train_dic)
            tot_loss += loss.item()
            pred_tr.extend(pred)
            gt_tr.extend(gt)
            loss.backward()
            optimizer.step()

        fusion_model.eval()
        aud_model.eval()
        text_model.eval()
        scheduler.step()
        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            pred_val = []
            gt_val = []
            for ind, val_dic in enumerate(val_loader):
                length = val_dic['length'].to(device)
                speaker_mask = val_dic['speaker_mask'].to(device)
                inp = val_dic['aud'].permute(0, 2, 1).double()
                val_dic['aud'], _ = aud_model(inp.to(device), length.to(device), speaker_mask)
                inp = val_dic['asr'].permute(0, 2, 1).double()
                val_dic['asr'], _ = text_model(inp.to(device), length.to(device), speaker_mask)
                _, val_out = fusion_model(val_dic['aud'], val_dic['asr'], val_dic['length'].to(device))
                #_, val_out = fusion_model(val_dic['asr'], val_dic['aud'], val_dic['length'].to(device))

                val_dic['target'][val_dic['target'] == -1] = 7
                pred, gt = compute_accuracy(val_out.cpu(), val_dic)
                pred_val.extend(pred)
                gt_val.extend(gt)
                val_loss += compute_loss(val_out.to(device), val_dic).item()
            if val_loss < final_val_loss:
                torch.save({'model_state_dict': fusion_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                            'asr_best_model.tar')
                #test_model('asr_best_model.tar')
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
