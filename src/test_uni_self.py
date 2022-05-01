import logging
from src.train_uni_self import MyDataset
from src.train_uni_self import compute_accuracy
from src.train_uni_self import GRUModel
from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM, GRU_DIM
from config import USE_TEXT, USE_AUDIO, unimodal_folder, USE_ASR
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.metrics import f1_score
import pandas as pd

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(PATH, test_aud, test_text, asr, test_labels, test_lengths, test_speaker_mask, max_length):
    
    test_dataset = MyDataset(test_aud, test_text, asr, test_labels,
                            test_lengths, test_speaker_mask)
    test_size = test_text.shape[0]
    indices = list(range(test_size))
    #test_sampler = SubsetRandomSampler(indices)
    
    #device = torch.device("cpu")
    test_loader = DataLoader(test_dataset,
                            batch_size=64,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False,
                            )
    test_results = dict()

    if USE_TEXT:
        test_model = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    if USE_AUDIO:
        test_model = GRUModel(AUDIO_DIM, GRU_DIM, 7, 0.5, 2, True).double()

    model_path = PATH
    checkpoint = torch.load(model_path)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.to(device)
    test_model.eval()
    pred_test = []
    gt_test = []
    for ind, test_dic in enumerate(test_loader):
        
        if USE_AUDIO:
            inp = test_dic['aud'].permute(0, 2, 1).double()
        if USE_TEXT:
            if USE_ASR:
                inp = test_dic['asr'].permute(0, 2, 1).double()
            else:
                inp = test_dic['text'].permute(0, 2, 1).double()
        length = test_dic['length']
        speaker_mask = test_dic['speaker_mask'].to(device)
        _, test_out = test_model(inp.to(device), length.to(device), speaker_mask)
        test_dic['target'][test_dic['target'] == -1] = 7
        pred, gt = compute_accuracy(test_out.cpu(), test_dic)
        pred_test.extend(pred)
        gt_test.extend(gt)
    pred_df = pd.DataFrame({'Predictions':pred_test})
    #pred_df.to_excel('"Predictions.xlsx', index = False)
    acc_fold = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Accuracy - {acc_fold}")
