from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM
from config import USE_TEXT, USE_AUDIO, LR, EPOCHS
from config import unimodal_folder, GRU_DIM
from config import test_npy_audio_path, asr_test_npy_text_path,test_npy_label_path, test_npy_text_path
from config import train_speaker_mask_path, test_speaker_mask_path
from config import length_test
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
from src.train_uni_asr import FusionModule, FusionModuleAll
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


def count_parameters(model):
    print("NUM PARAMETERS", sum(p.numel() for p in model.parameters() if p.requires_grad))


def test_model(PATH):
    test_aud = np.load(test_npy_audio_path)
    test_text = np.load(test_npy_text_path)
    test_labels = np.load(test_npy_label_path)
    test_lengths = np.load(length_test)
    max_length = test_lengths.shape[1]
    asr = np.load(asr_test_npy_text_path)
    speaker_mask = np.load(test_speaker_mask_path)

    test_dataset = MyDataset(test_aud, test_text, asr, test_labels,
                            test_lengths, speaker_mask)
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


    test_model = FusionModuleAll(400, 400*4, NUM_HIDDEN_LAYERS)
    #test_model = FusionModule_spe(GRU_DIM*4, GRU_DIM*2, NUM_HIDDEN_LAYERS)
    model_path = PATH
    checkpoint = torch.load(model_path)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.to(device)
    test_model.eval()

    text_model = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    text_model.to(device)
    checkpoint_text = torch.load('text_self_best_model.tar')
    text_model.load_state_dict(checkpoint_text['model_state_dict'])
    text_model.eval()

    aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    aud_model.to(device)
    checkpoint_aud = torch.load('aud_self_best_model.tar')
    aud_model.load_state_dict(checkpoint_aud['model_state_dict'])
    aud_model.eval()

    text_model_asr = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    text_model_asr.to(device)
    checkpoint_text = torch.load('asr_text_self_best_model.tar')
    text_model_asr.load_state_dict(checkpoint_text['model_state_dict'])
    text_model_asr.eval()

    fusion_model = FusionModule(400*4, 2).double()
    #fusion_model = FusionModuleAll(GRU_DIM*2, GRU_DIM*2, NUM_HIDDEN_LAYERS)
    fusion_model.to(device)
    checkpoint = torch.load('asr_best_model.tar')
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    fusion_model.eval()

    pred_test, gt_test = [], []

    for ind, test_dic in enumerate(test_loader):
        length = test_dic['length']
        speaker_mask = test_dic['speaker_mask'].to(device)
        inp = test_dic['aud'].permute(0, 2, 1).double()
        test_dic['aud'], _ = aud_model.forward(inp.to(device), length.to(device), speaker_mask)
        inp = test_dic['text'].permute(0, 2, 1).double()
        test_dic['text'], _ = text_model.forward(inp.to(device), length.to(device), speaker_mask)
        inp_asr = test_dic['asr'].permute(0, 2, 1).double()
        asr, _ = text_model_asr(inp_asr.to(device), length.to(device), speaker_mask)

        fused, _ = fusion_model(test_dic['aud'], asr, length.to(device))
        #fused, _ = fusion_model(asr, test_dic['aud'], length.to(device))
        _, test_out = test_model(test_dic['text'], fused, test_dic['length'].to(device))
        test_dic['target'][test_dic['target'] == -1] = 7
        pred, gt = compute_accuracy(test_out.cpu(), test_dic)
        pred_test.extend(pred)
        gt_test.extend(gt)
    
    acc_f1 = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Accuracy of fold- {acc_f1}")


def train(train_data, val_data, max_length):
    base_lr = LR
    n_gpu = torch.cuda.device_count()
    

    final_val_loss = 999999
    
    train_loader = train_data
    val_loader = val_data

    model = FusionModuleAll(400, 400*4, NUM_HIDDEN_LAYERS)
    #model = FusionModule_spe(GRU_DIM*4, GRU_DIM*2, NUM_HIDDEN_LAYERS)
    model.to(device)
    count_parameters(model)

    aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    text_model = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 2, True).double()

    #text_model = GRUModel(TEXT_DIM, GRU_DIM, 4, 0.5, 3, True).double()
    text_model.to(device)
    checkpoint_text = torch.load('text_self_best_model.tar')
    text_model.load_state_dict(checkpoint_text['model_state_dict'])
    
    aud_model.to(device)
    checkpoint_aud = torch.load('aud_self_best_model.tar')
    aud_model.load_state_dict(checkpoint_aud['model_state_dict'])
    

    text_model_asr = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    text_model_asr.to(device)
    checkpoint_text = torch.load('asr_text_self_best_model.tar')
    text_model_asr.load_state_dict(checkpoint_text['model_state_dict'])
    
    fusion_model = FusionModule(400*4, 2).double()
    #fusion_model = FusionModuleAll(GRU_DIM*2, GRU_DIM*2, NUM_HIDDEN_LAYERS)
    fusion_model.to(device)
    checkpoint = torch.load('asr_best_model.tar')
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    '''
    for param in text_model.parameters():
        param.requires_grad = False
    for param in aud_model.parameters():
        param.requires_grad = False
    for param in text_model_asr.parameters():
        param.requires_grad = False
    for param in fusion_model.parameters():
        param.requires_grad = False'''
    text_model.eval()
    aud_model.eval()
    fusion_model.eval()
    text_model_asr.eval()
    
    optimizer = Adam(model.parameters(), lr=base_lr)
    scheduler = MultiStepLR(optimizer, milestones=[50,175], gamma=0.1)
    #scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    for e in range(EPOCHS):
        print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))
        tot_loss, tot_acc = 0.0, 0.0
        model.train()
        pred_tr = []
        gt_tr = []
        
        for ind, train_dic in enumerate(train_loader):
            model.zero_grad()
            length = train_dic['length'].to(device)
            speaker_mask = train_dic['speaker_mask'].to(device)

            inp = train_dic['aud'].permute(0, 2, 1).double()
            train_dic['aud'], _ = aud_model(inp.to(device), length.to(device), speaker_mask)

            inp = train_dic['text'].permute(0, 2, 1).double()
            train_dic['text'], _ = text_model(inp.to(device), length.to(device), speaker_mask)

            inp_asr = train_dic['asr'].permute(0, 2, 1).double()
            asr, _ = text_model_asr(inp_asr.to(device), length.to(device), speaker_mask)

            fused, _ = fusion_model(train_dic['aud'], asr, length.to(device))
            #fused, _ = fusion_model(asr, train_dic['aud'], length.to(device))

            _, out = model(train_dic['text'], fused, train_dic['length'].to(device))
            #torch.save(out1, 'gpu_out'+str(e)+str(ind)+".pt")
            train_dic['target'][train_dic['target'] == -1] = 7
            pred, gt = compute_accuracy(out.cpu(), train_dic)
            loss = compute_loss(out.to(device), train_dic)
            tot_loss += loss.item()
            pred_tr.extend(pred)
            gt_tr.extend(gt)
            loss.backward()
            optimizer.step()

        model.eval()
        text_model.eval()
        aud_model.eval()
        fusion_model.eval()
        text_model_asr.eval()
        #scheduler.step()
        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            pred_val = []
            gt_val = []
            for ind, val_dic in enumerate(val_loader):
                length = val_dic['length'].to(device)
                speaker_mask = val_dic['speaker_mask'].to(device)

                inp = val_dic['aud'].permute(0, 2, 1).double()
                val_dic['aud'], _ = aud_model.forward(inp.to(device), length.to(device), speaker_mask)

                inp = val_dic['text'].permute(0, 2, 1).double()
                val_dic['text'], _ = text_model.forward(inp.to(device), length.to(device), speaker_mask)

                inp_asr = val_dic['asr'].permute(0, 2, 1).double()
                asr, _ = text_model_asr(inp_asr.to(device), length.to(device), speaker_mask)

                fused, _ = fusion_model(val_dic['aud'], asr, length.to(device))
                #fused, _ = fusion_model(asr, val_dic['aud'], length.to(device))
                _, val_out = model(val_dic['text'], fused, val_dic['length'].to(device))

                val_dic['target'][val_dic['target'] == -1] = 7
                pred, gt = compute_accuracy(val_out.cpu(), val_dic)
                pred_val.extend(pred)
                gt_val.extend(gt)
                val_loss += compute_loss(val_out.to(device), val_dic).item()
            if val_loss < final_val_loss:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                            'asr_cross_best_model.tar')
                test_model('asr_cross_best_model.tar')
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
