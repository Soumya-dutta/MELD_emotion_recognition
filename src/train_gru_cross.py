from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM, IMAGE_DIM
from config import USE_TEXT, USE_AUDIO, USE_IMAGE, LR, EPOCHS
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
from src.train_ca import FusionModule
from src.train_uni_self import MyDataset, compute_accuracy, compute_loss, GRUModel


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
#device = torch.device("cpu")


def count_parameters(model):
    print("NUM PARAMETERS", sum(p.numel() for p in model.parameters() if p.requires_grad))

def train(train_data, val_data, max_length):
    base_lr = LR
    n_gpu = torch.cuda.device_count()
    for fold in range(1):
        final_val_loss = 999999
        logger.info(f"Running fold {fold}")
        train_loader = train_data[fold]
        val_loader = val_data[fold]
        model = FusionModule(GRU_DIM*2, NUM_HIDDEN_LAYERS)
        model.to(device)
        checkpoint = torch.load('cross_best_model_' + str(fold) + '.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        count_parameters(model)
        if USE_TEXT:
            text_model = GRUModel(TEXT_DIM, GRU_DIM, 4, 0.5, 3, True).double()
            text_model.to(device)
            checkpoint_text = torch.load(os.path.join(unimodal_folder, 'best_model_text'+str(fold)+'.tar'))
            text_model.load_state_dict(checkpoint_text['model_state_dict'])
        if USE_AUDIO:
            aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 4, 0.5, 2, True).double()
            aud_model.to(device)
            checkpoint_aud = torch.load(os.path.join(unimodal_folder, 'best_model_aud'+str(fold)+'.tar'))
            aud_model.load_state_dict(checkpoint_aud['model_state_dict'])


        optimizer = Adam([{'params':model.parameters()},
                          {'params':text_model.parameters()},
                          {'params':aud_model.parameters()}], lr=base_lr)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
        for e in range(EPOCHS):
            print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))
            tot_loss, tot_acc = 0.0, 0.0
            model.train()
            text_model.train()
            aud_model.train()
            for ind, train_dic in enumerate(train_loader):
                model.zero_grad()
                text_model.zero_grad()
                aud_model.zero_grad()
                length = train_dic['length']
                if USE_AUDIO:
                    inp = train_dic['aud'].permute(0, 2, 1).double()
                    train_dic['aud'], _ = aud_model(inp.to(device), length.to(device))
                if USE_TEXT:
                    inp = train_dic['text'].permute(0, 2, 1).double()
                    train_dic['text'], _ = text_model.forward(inp.to(device), length.to(device))

                out = model(train_dic['text'], train_dic['aud'], train_dic['length'].to(device))
                #torch.save(out1, 'gpu_out'+str(e)+str(ind)+".pt")
                train_dic['target'][train_dic['target'] == -1] = 4
                acc = compute_accuracy(out.cpu(), train_dic)
                loss = compute_loss(out.to(device), train_dic)
                tot_loss += loss.item()
                tot_acc += acc.item()
                loss.backward()
                optimizer.step()
            model.eval()
            text_model.eval()
            aud_model.eval()
            #scheduler.step()
            with torch.no_grad():
                val_loss, val_acc = 0.0, 0.0
                for ind, val_dic in enumerate(val_loader):
                    length = val_dic['length']
                    if USE_AUDIO:
                        inp = val_dic['aud'].permute(0, 2, 1).double()
                        val_dic['aud'], _ = aud_model.forward(inp.to(device), length.to(device))
                    if USE_TEXT:
                        inp = val_dic['text'].permute(0, 2, 1).double()
                        val_dic['text'], _ = text_model.forward(inp.to(device), length.to(device))

                    val_out = model(val_dic['text'], val_dic['aud'], val_dic['length'].to(device))
                    val_dic['target'][val_dic['target'] == -1] = 4
                    val_acc += compute_accuracy(val_out.cpu(), val_dic).item()
                    val_loss += compute_loss(val_out.to(device), val_dic).item()
                if val_loss < final_val_loss:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                'ee_cross_best_model_' + str(fold) + '.tar')
                    torch.save({'model_state_dict': text_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                'text_best_model_ee' + str(fold) + '.tar')
                    torch.save({'model_state_dict': aud_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                'audio_best_model_ee' + str(fold) + '.tar')
                    final_val_loss = val_loss
                e_log = e + 1
                train_loss = tot_loss/len(train_loader)
                train_acc = tot_acc/len(train_loader)
                val_loss_log = val_loss/len(val_loader)
                val_acc_log = val_acc/len(val_loader)
                logger.info(f"Epoch {e_log}, \
                            Training Loss {train_loss},\
                            Training Accuracy {train_acc}")
                logger.info(f"Epoch {e_log}, \
                            Validation Loss {val_loss_log},\
                            Validation Accuracy {val_acc_log}")
