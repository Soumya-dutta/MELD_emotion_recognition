pickle_path_asr = "features/train_roberta_wav2vec_asr_improved.pkl"
pickle_path = "features/train_roberta_wav2vec.pkl"

length_train = "features/length_train.npy"
length_test = "features/length_test.npy"

train_npy_text_path = "features/train_text.npy"
asr_train_npy_text_path = "features/train_text_asr.npy"
train_npy_audio_path = "features/train_audio.npy"
train_npy_label_path = "features/train_labels.npy"
train_text_csv_path = "features/train_text.csv"
train_speaker_mask_path = "features/train_speaker_mask.npy"
asr_train_npy_text_path = "features/asr_train_text.npy"

test_npy_audio_path = "features/test_audio.npy"
test_npy_text_path = "features/test_text.npy"
asr_test_npy_text_path = "features/test_text_asr.npy"
test_npy_label_path = "features/test_labels.npy"
test_text_csv_path = "features/test_text.csv"
test_speaker_mask_path = "features/test_speaker_mask.npy"
asr_test_npy_text_path = "features/asr_test_text.npy"

unimodal_folder = "unimodal_models/"

NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 3
HIDDEN_SIZE = 60
AUDIO_DIM = 768
TEXT_DIM = 768

GRU_DIM = 200

USE_TEXT = True
USE_AUDIO = True

USE_CROSSATTEN = True
USE_GRU_CROSS = False
USE_SELF_ATTEN = False
USE_ASR = True

LR = 1e-5
EPOCHS = 50
