import logging
from src.train_uni_self import GRUModel, MyDataset, compute_accuracy
from src.train_uni_asr import FusionModule
from src.train_uni_asr import FusionModuleAll
from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM, GRU_DIM
from config import USE_TEXT, USE_AUDIO, unimodal_folder
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.metrics import f1_score
from src.train_uni_self import compute_loss

torch.backends.cudnn.enabled = False
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def test_model(PATH, test_aud, test_text, asr, test_labels, test_lengths, test_speaker_mask, max_length):
    test_dataset = MyDataset(test_aud, test_text, asr, test_labels,
                            test_lengths, test_speaker_mask)
    test_size = test_aud.shape[0]
    indices = list(range(test_size))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    test_loader = DataLoader(test_dataset,
                            batch_size=32,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False,
                            )
    test_results = dict()

    test_model = FusionModuleAll(GRU_DIM*4, GRU_DIM*8, NUM_HIDDEN_LAYERS)
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

    fusion_model = FusionModule(GRU_DIM*8, 3).double()
    #fusion_model = FusionModuleAll(GRU_DIM*2, GRU_DIM*2, NUM_HIDDEN_LAYERS)
    fusion_model.to(device)
    checkpoint = torch.load('asr_best_model.tar')
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    fusion_model.eval()

    pred_test = []
    gt_test = []
    tot_loss = 0.0

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
        loss = compute_loss(test_out.to(device), test_dic)
        tot_loss += loss.item()
        pred_test.extend(pred)
        gt_test.extend(gt)

    acc_fold = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Accuracy - {acc_fold}")
    print(tot_loss/len(test_loader))
