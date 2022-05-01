import numpy as np
import logging
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from src.read_data import create_data
from src.train_uni_self import MyDataset
from src.train_uni_self import train as train_uni
from src.train_uni_asr import train as train_uni_asr
from src.train_ca_asr import train as train_ca_asr
from src.test_uni_self import test_model as test_uni
from src.test_uni_asr import test_model as test_uni_asr
from src.test_ca_asr import test_model as test_ca_asr
from src.train_ca import train as train_ca
from src.test_ca import test_model as test_ca
'''
from src.train_gru_cross import train as train_gru_cross
from src.test_gru_cross import test_model as test_gru_cross
'''

from config import train_npy_audio_path, train_npy_text_path
from config import train_npy_label_path, pickle_path_asr, pickle_path
from config import test_npy_audio_path, asr_test_npy_text_path, asr_train_npy_text_path
from config import test_npy_label_path, test_npy_text_path
from config import length_train, length_test, train_text_csv_path
from config import test_text_csv_path, USE_AUDIO, USE_TEXT, USE_ASR
from config import USE_CROSSATTEN, USE_GRU_CROSS
from config import USE_SELF_ATTEN
from config import train_speaker_mask_path, test_speaker_mask_path

SEED = 4213
torch.manual_seed(SEED)
np.random.seed(SEED)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
global device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

def create_train_data(asr):
    train_data = dict()
    val_data = dict()
    batch_size = 32
        
    train_aud = np.load(train_npy_audio_path)
    train_text_asr = np.load(asr_train_npy_text_path)
    train_text = np.load(train_npy_text_path)
    train_labels = np.load(train_npy_label_path)
    train_lengths = np.load(length_train)
    train_speaker_mask = np.load(train_speaker_mask_path)    
    val_aud = np.load(train_npy_audio_path.replace('train', 'val'))
    val_text_asr = np.load(asr_train_npy_text_path.replace('train', 'val'))
    val_text = np.load(train_npy_text_path.replace('train', 'val'))
    val_labels = np.load(train_npy_label_path.replace('train', 'val'))
    val_lengths = np.load(length_train.replace('train', 'val'))
    val_speaker_mask = np.load(train_speaker_mask_path.replace('train', 'val'))

    train_size, val_size = train_text.shape[0], val_text.shape[0]
    train_indices, val_indices = list(range(train_size)), list(range(val_size))
    #train_sampler = SubsetRandomSampler(train_indices)
    #valid_sampler = SubsetRandomSampler(val_indices)
    
    train_dataset = MyDataset(train_aud, train_text, train_text_asr, train_labels,
                            train_lengths, train_speaker_mask)
    val_dataset = MyDataset(val_aud, val_text, val_text_asr, val_labels,
                            val_lengths, val_speaker_mask)
    '''train_dataset = MyDataset([], train_text, [], train_labels,
                            train_lengths)
    val_dataset = MyDataset([], val_text, [], val_labels,
                            val_lengths)'''                        
    train_loader = DataLoader(train_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False,
                    )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False
                            )

    return train_loader, val_loader, val_lengths.shape[1]

def main(create_flag, train_flag, test_flag):
    '''Depending on the three input flags it creates the data, trains a model
    based on the same and then tests the trained model.
    '''
    if create_flag:
        create_data(pickle_path, train_npy_audio_path, train_npy_text_path,
                    train_npy_label_path, test_npy_audio_path, test_npy_text_path,
                    test_npy_label_path, length_train, length_test, 
                    train_text_csv_path, test_text_csv_path)
        if USE_ASR:
            create_data(pickle_path_asr, train_npy_audio_path, asr_train_npy_text_path,
                    train_npy_label_path, test_npy_audio_path, 
                    asr_test_npy_text_path, test_npy_label_path, length_train, length_test, 
                    train_text_csv_path, test_text_csv_path)
    if train_flag:
        train_data, val_data, max_length = create_train_data(False)
        if USE_ASR:
            train_data_asr, val_data_asr, _ = create_train_data(True)
        if USE_SELF_ATTEN:
            if USE_ASR:
                train_uni(train_data_asr, val_data_asr, max_length)
            else:
                train_uni(train_data, val_data, max_length)
        if USE_CROSSATTEN:
            if USE_AUDIO and not USE_TEXT and USE_ASR:
                train_uni_asr(train_data_asr, val_data_asr, max_length)
            if USE_AUDIO and USE_TEXT and USE_ASR:
                train_ca_asr(train_data, val_data, max_length)
        if USE_AUDIO and USE_TEXT and not USE_ASR:
            train_ca(train_data, val_data, max_length)


    if test_flag:

        test_aud = np.load(test_npy_audio_path)
        test_text = np.load(test_npy_text_path)
        test_labels = np.load(test_npy_label_path)
        test_lengths = np.load(length_test)
        max_length = test_lengths.shape[1]
        test_speaker_mask = np.load(test_speaker_mask_path)
        test_text_asr = np.load(asr_test_npy_text_path)
        if USE_SELF_ATTEN:
            if USE_AUDIO:
                if USE_ASR:
                    test_uni('asr_aud_self_best_model.tar', test_aud, test_text, test_text_asr, test_labels,
                        test_lengths, test_lengths.shape[1])
                else:
                    test_uni('aud_self_best_model.tar', test_aud, test_text, test_text_asr, test_labels,
                        test_lengths, test_speaker_mask, test_lengths.shape[1])
            if USE_TEXT: 
                if USE_ASR:
                    test_uni('asr_text_self_best_model.tar', test_aud, test_text, test_text_asr, test_labels,
                        test_lengths, test_speaker_mask, test_lengths.shape[1])
                else:
                    test_uni('text_self_best_model.tar', test_aud, test_text, test_text_asr, test_labels,
                        test_lengths, test_speaker_mask, test_lengths.shape[1])
        if USE_CROSSATTEN:
            if USE_AUDIO and not USE_TEXT and USE_ASR:
                test_uni_asr('asr_best_model.tar', test_aud, test_text, test_text_asr, test_labels,
                        test_lengths, test_speaker_mask, test_lengths.shape[1])
            if USE_AUDIO and USE_TEXT and USE_ASR:
                test_ca_asr('asr_cross_best_model.tar', test_aud, test_text, test_text_asr, test_labels,
                        test_lengths, test_speaker_mask, test_lengths.shape[1])
        if USE_AUDIO and USE_TEXT and not USE_ASR:
            test_ca('cross_best_model.tar', test_aud, test_text, test_text_asr, test_labels,
                    test_lengths, test_speaker_mask, test_lengths.shape[1])

if __name__ == "__main__":
    main(True, True, True)

