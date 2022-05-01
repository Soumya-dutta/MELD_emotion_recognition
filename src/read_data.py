import pickle5 as pickle
import numpy as np
import pandas as pd
from config import TEXT_DIM, AUDIO_DIM
from config import train_speaker_mask_path, test_speaker_mask_path

def create_test_mask(videos, labels, max_utt_length, file_name):
    mask = np.empty([len(videos), max_utt_length])
    for ind, vid in enumerate(videos):
        act_length = len(labels[vid])
        rem_length = max_utt_length - act_length
        ones = np.ones((act_length, 1))
        zeros = np.zeros((rem_length, 1))
        total = np.concatenate((ones, zeros), axis = 0)
        mask[ind, :] = total[:, 0]
    with open(file_name, 'wb') as f:
        np.save(f, mask)

def create_mask(labels, max_utt_length, file_name):
    num_vid = len(labels)
    mask = np.empty([num_vid, max_utt_length])
    #videos = sorted(labels, key=lambda k:len(labels[k]))
    videos = list(labels.keys())
    for ind, vid in enumerate(videos):
        act_length = len(labels[vid])
        rem_length = max_utt_length - act_length
        ones = np.ones((act_length, 1))
        zeros = np.zeros((rem_length, 1))
        total = np.concatenate((ones, zeros), axis = 0)
        mask[ind, :] = total[:, 0]
    with open(file_name, 'wb') as f:
        #mask_train = np.concatenate((mask_train, mask_val), axis = 0)
        np.save(f, mask)

def create_test_array(videos, max_utt_length, mat, feature_length, mode, file_name):
    if mode == 'label':
        Y = np.empty([len(videos), max_utt_length])
    else:
        Y = np.empty([len(videos), feature_length, max_utt_length])
    for ind, vid in enumerate(videos):
        if mode == 'label':
            target = np.array(mat[vid]).reshape(-1, 1)
            rem_length = max_utt_length - target.shape[0]
            mask = -1*np.ones((rem_length, 1))
            target = np.concatenate((target, mask), axis = 0)
            target[target == 4] = 0
            Y[ind, :] = target[:,0]
        else:
            target = np.array(mat[vid]).T
            rem_length = max_utt_length - target.shape[1]
            mask = np.zeros((feature_length, rem_length))
            target = np.concatenate((target, mask), axis = 1)
            Y[ind, :, :] = target
    with open(file_name, 'wb') as f:
        np.save(f, Y)

def create_array(max_utt_length, mat, feature_length, mode, file_name):
    num = len(mat)

    #videos = sorted(mat, key=lambda k:len(mat[k]))
    videos = list(mat.keys())
    if mode == 'label':
        Y = np.empty([num, max_utt_length])  
    else:
        Y = np.empty([num, feature_length, max_utt_length])
    for ind, vid in enumerate(videos):
        if mode == 'label':
            target = np.array(mat[vid]).reshape(-1, 1)
            rem_length = max_utt_length - target.shape[0]
            mask = -1*np.ones((rem_length, 1))
            target = np.concatenate((target, mask), axis = 0)
            Y[ind, :] = target[:,0]
        else:
            target = np.array(mat[vid]).T
            rem_length = max_utt_length - target.shape[1]
            mask = np.zeros((feature_length, rem_length))
            target = np.concatenate((target, mask), axis = 1)
            Y[ind, :, :] = target
    print(Y.shape)
    with open(file_name, 'wb') as f:
        #Y_train = np.concatenate((Y_train, Y_val), axis = 0)
        np.save(f, Y)    

def create_text_csv(videos, labels, sentences, path):
    df = pd.DataFrame(columns = ['Vid_name','Utterance', 'Emotion'])
    for vid_id in videos:
        utterance_list = sentences[vid_id]
        emotion_list = labels[vid_id]
        df2 = pd.DataFrame()
        df2['Utterance'] = utterance_list
        df2['Emotion'] = emotion_list
        df2['Vid_name'] = [vid_id]*len(emotion_list)
        df = df.append(df2, ignore_index = True) 
    df["Emotion"] = df["Emotion"].replace(4, 0)
    df = df[df["Emotion"] < 4]
    df.to_csv(path, index=False)

def get_attention_mask(speakers, max_utt_length, path):
    #videos = sorted(speakers, key=lambda k:len(speakers[k]))
    videos = list(speakers.keys())
    speaker_mask = np.zeros((len(videos), max_utt_length, max_utt_length))
    for i, vid in enumerate(videos):
        speakers_conv = speakers[vid]
        speaker_ohe = pd.get_dummies(speakers_conv)
        speaker_ohe = speaker_ohe.to_dict()
        speaker_ohe = {k : list(v.values()) for k, v in speaker_ohe.items()}
        length = len(speakers_conv)
        for ind, speaker in enumerate(speakers_conv):
            speaker_mask[i, ind, :length] = speaker_ohe[speaker]
    with open(path, 'wb') as f:
        np.save(f, speaker_mask)    

def create_data(pickle_path, train_aud, train_text, train_lab, test_aud,
                test_text, test_lab, path_train, path_test, train_csv, test_csv):

    val_pickle_path = pickle_path.replace("train", "val")
    test_pickle_path = pickle_path.replace("train", "test")

    f_train = pickle.load(open(pickle_path, 'rb'), encoding= 'latin1')
    f_val = pickle.load(open(val_pickle_path, 'rb'), encoding= 'latin1')
    f_test = pickle.load(open(test_pickle_path, 'rb'), encoding= 'latin1')
    labels_train, text_train, audio_train = f_train[2], f_train[3], f_train[4]
    labels_val, text_val, audio_val= f_val[2], f_val[3], f_val[4]
    labels_test, text_test, audio_test= f_test[2], f_test[3], f_test[4]
    if 'asr' in pickle_path:
        text_train = f_train[-1]
        text_val = f_val[-1]
        text_test = f_test[-1]
    #sentences, train_videos, test_videos = f[6], f[7], f[8]

    max_utt_length_tr = max(len(labels_train[i]) for i in labels_train)
    max_utt_length_va = max(len(labels_val[i]) for i in labels_val)
    max_utt_length_te = max(len(labels_test[i]) for i in labels_test)
    max_utt_length = max(max_utt_length_te, max_utt_length_tr, max_utt_length_va)


    audio_feature_length = AUDIO_DIM
    text_feature_length = TEXT_DIM
    create_mask(labels_train, max_utt_length, path_train)
    path_val = path_train.replace("train", "val")
    create_mask(labels_val, max_utt_length, path_val)
    create_mask(labels_test, max_utt_length, path_test)

    create_array(max_utt_length, labels_train, 1, 'label', train_lab)
    val_lab = train_lab.replace("train", "val")
    create_array(max_utt_length, labels_val, 1, 'label', val_lab)
    create_array(max_utt_length, labels_test, 1, 'label', test_lab)

    create_array(max_utt_length, text_train, text_feature_length, 'text', train_text)
    val_text = train_text.replace("train", "val")
    create_array(max_utt_length, text_val, text_feature_length, 'text', val_text)
    create_array(max_utt_length, text_test, text_feature_length, 'text', test_text)
    #test_text: target path of text npy
    #text_text: array read from pickle
    get_attention_mask(f_train[1], max_utt_length, train_speaker_mask_path)
    val_speaker_mask_path = train_speaker_mask_path.replace("train", "val")
    get_attention_mask(f_val[1], max_utt_length, val_speaker_mask_path)
    get_attention_mask(f_test[1], max_utt_length, test_speaker_mask_path)

    create_array(max_utt_length, audio_train, audio_feature_length, 'audio', train_aud)
    val_aud = train_aud.replace("train", "val")
    create_array(max_utt_length, audio_val, audio_feature_length, 'audio', val_aud)
    create_array(max_utt_length, audio_test, audio_feature_length, 'audio', test_aud)
    '''
    train_videos = list(train_videos)
    train_videos.sort()
    sessions = ['Ses01', 'Ses02', 'Ses03', 'Ses04']
    for ind, ses_no in enumerate(sessions):
        create_mask(train_videos, labels, max_utt_length, path_train, ses_no, ind)
        create_array(train_videos, max_utt_length, labels, 1, 'label', train_lab, ses_no, ind)
        create_array(train_videos, max_utt_length, audio, audio_feature_length, 'audio', train_aud, ses_no, ind)
        create_array(train_videos, max_utt_length, text, text_feature_length, 'text', train_text, ses_no, ind)
        #create_array(train_videos, max_utt_length, image, image_feature_length, 'image', train_img, ses_no, ind)
    create_text_csv(train_videos, labels, sentences, train_csv)

    test_videos = list(test_videos)
    test_videos.sort()
    create_test_mask(test_videos, labels, max_utt_length, path_test)
    create_test_array(test_videos, max_utt_length, labels, 1, 'label', test_lab)
    create_test_array(test_videos, max_utt_length, audio, audio_feature_length, 'audio', test_aud)
    create_test_array(test_videos, max_utt_length, text, text_feature_length, 'text', test_text)
    #create_test_array(test_videos, max_utt_length, image, image_feature_length, 'image', test_img)
    create_text_csv(test_videos, labels, sentences, test_csv)'''
