import logging
from src.train_uni_self import GRUModel, MyDataset, compute_accuracy
from src.train_uni_asr import FusionModule
from src.train_ca_asr import FusionModuleAll
from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM, GRU_DIM
from config import USE_TEXT, USE_AUDIO, unimodal_folder
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.metrics import f1_score

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
    test_sampler = SubsetRandomSampler(indices)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    test_loader = DataLoader(test_dataset,
                            sampler = test_sampler,
                            batch_size=64,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False,
                            )
    test_results = dict()
    test_model = FusionModule(400*4, 2).double()
    model_path = PATH
    checkpoint = torch.load(model_path)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.to(device)
    test_model.eval()

    aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    aud_model.to(device)
    checkpoint_aud = torch.load(os.path.join('aud_self_best_model.tar'))
    aud_model.load_state_dict(checkpoint_aud['model_state_dict'])
    aud_model.eval()

    text_model = GRUModel(TEXT_DIM, GRU_DIM, 7, 0.5, 2, True).double()
    text_model.to(device)
    checkpoint_text = torch.load(os.path.join('asr_text_self_best_model.tar'))
    text_model.load_state_dict(checkpoint_text['model_state_dict'])
    text_model.eval()
    pred_test = []
    gt_test = []
    for ind, test_dic in enumerate(test_loader):
        length = test_dic['length']
        speaker_mask = test_dic['speaker_mask'].to(device)
        inp = test_dic['aud'].permute(0, 2, 1).double()
        test_dic['aud'], _ = aud_model.forward(inp.to(device), length.to(device), speaker_mask)
        inp = test_dic['asr'].permute(0, 2, 1).double()
        test_dic['asr'], _ = text_model.forward(inp.to(device), length.to(device), speaker_mask)
        _, test_out = test_model.forward(test_dic['aud'], test_dic['asr'], test_dic['length'].to(device))
        test_dic['target'][test_dic['target'] == -1] = 7

        pred, gt = compute_accuracy(test_out.cpu(), test_dic)
        pred_test.extend(pred)
        gt_test.extend(gt)
    acc_fold = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Accuracy - {acc_fold}")
