# Main file for running the code

from config import Config
from mimic3models import common_utils
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.in_hospital_mortality import utils as ihm_utils
'''
In intensive care units, where patients come in with a wide range of health conditions, 
triaging relies heavily on clinical judgment. ICU staff run numerous physiological tests, 
such as bloodwork and checking vital signs, 
to determine if patients are at immediate risk of dying if not treated aggressively.
'''

import utils
from Models import OnlyTextModel, ClinicalNotesModel
import pickle
import numpy as np
import sys
import random
import os

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score

from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

import warnings
import time

start_time = time.time()
# Ignoring warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

conf = utils.get_config() # Get configs
args = utils.get_args() # Get arguments

if args["Seed"]:
    torch.manual_seed(args["Seed"])
    np.random.seed(args["Seed"])


# Loading pre-trained model based on EmbedModel argument
if args['model_name'] == 'BioBert':
    if args["EmbedModel"] == "bioRoberta":
        EmbedModelName = "bioRoberta"
        tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
        BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")
    elif args["EmbedModel"] == "BioBert":
        EmbedModelName = "BioBert"
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        BioBert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif args["EmbedModel"] == "Bert":
        EmbedModelName = "SimpleBert"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BioBert = BertModel.from_pretrained("bert-base-uncased")
    BioBert = BioBert.to(device)
    BioBertConfig = BioBert.config
    # conf.max_len = BioBertConfig.max_position_embeddings
    W_emb = None
    lookup = None
    word2index_lookup = None
    
# Loading BioWordVec word embeddings
else:
    vectors, word2index_lookup = utils.get_embedding_dict(conf)
    W_emb = torch.tensor(vectors)
    
    lookup = utils.lookup
    # let's set pad token to zero padding instead of random padding. This is for word2vec model
    if conf.padding_type == 'Zero':
        print("Zero Padding..")
        vectors[lookup(word2index_lookup, '<pad>')] = 0
    BioBert = None
    BioBertConfig = None
print(str(vars(conf)))
print(str(args))

number_epoch = int(args['number_epoch'])
batch_size = int(args['batch_size'])
model_name = args['model_name']
model_type = args['model_type']

# We use checkpoint path as model name
ModelNameOut = args["checkpoint_path"]
dropout_keep_prob = conf.dropout

if (args['model_name'] == 'BioBert') and (args['TextModelCheckpoint'] != None):
    # Use text model checkpoint to update the bert model to fine-tuned weights
    OTmodel = OnlyTextModel(device, BioBert, BioBertConfig)
    OTmodel = OTmodel.to(device)
    checkpoint = torch.load(args['TextModelCheckpoint']) #--TextModelCheckpoint BBOnlyTextshort
    OTmodel.load_state_dict(checkpoint['state_dict'])
    BioBert = OTmodel.BioBert
    del OTmodel
    del checkpoint

if (args['model_name'] == 'BioBert') and bool(int(args['freeze_model'])):
    for param in BioBert.embeddings.parameters():
        param.requires_grad = False
    for param in BioBert.encoder.parameters():
        param.requires_grad = False


def init_all(model, init_func, *params, **kwargs):
    # Initialize all weights using init_func
    for p in model.parameters():
        init_func(p, *params, **kwargs)
        
def Evaluate(Labels, Preds, PredScores):
    # Get the evaluation metrics like AUC, percision and etc.
    percision, recall, fscore, support = precision_recall_fscore_support(Labels, Preds, average='binary')
    _, _, fscore_weighted, _ = precision_recall_fscore_support(Labels, Preds, average='weighted')
    accuracy = accuracy_score(Labels, Preds) * 100
    confmat = confusion_matrix(Labels, Preds)
    roc_macro, roc_micro, roc_weighted = roc_auc_score(Labels, PredScores, average='macro'), roc_auc_score(Labels, PredScores, average='micro'), roc_auc_score(Labels, PredScores, average='weighted')
    prf_test = {'percision': percision, 'recall': recall, 'fscore': fscore, 'fscore_weighted': fscore_weighted, 'accuracy': accuracy, 'confusionMatrix': confmat, 'roc_macro': roc_macro, 'roc_micro': roc_micro, 'roc_weighted': roc_weighted}
    return prf_test

def Evaluate_voting(Labels, Preds, PredScores, names):
    # Evaluate for flat data, taking ensemble of all the notes results available
    df = pd.DataFrame({'Labels': Labels,
               'Preds': Preds,
               'PredScores': PredScores,
               'names': names})
    df['Preds'] = df['Preds'] * 1 # Convert boolean to int
    df = df.groupby(['names']).mean()
    df['Preds'] = df['Preds'].round(0)
    Preds = df['Preds']
    PredScores = df['PredScores']
    Labels = df['Labels']
    # percision, recall, fscore, support = precision_recall_fscore_support(df['Labels'], Preds, average='binary')
    percision, recall, fscore, support = precision_recall_fscore_support(Labels, Preds, average='binary')
    _, _, fscore_weighted, _ = precision_recall_fscore_support(Labels, Preds, average='weighted')
    accuracy = accuracy_score(Labels, Preds) * 100
    confmat = confusion_matrix(Labels, Preds)
    roc_macro, roc_micro, roc_weighted = roc_auc_score(Labels, PredScores, average='macro'), roc_auc_score(Labels, PredScores, average='micro'), roc_auc_score(Labels, PredScores, average='weighted')
    prf_test = {'percision': percision, 'recall': recall, 'fscore': fscore, 'fscore_weighted': fscore_weighted, 'accuracy': accuracy, 'confusionMatrix': confmat, 'roc_macro': roc_macro, 'roc_micro': roc_micro, 'roc_weighted': roc_weighted}
    return prf_test

def Evaluate_Model(model, batch, names):
    # Test the model
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        FirstTime = False
        
        aucpr_obj = utils.AUCPR()
        
        PredScores = None
        for idx, sample in enumerate(batch):
            X = torch.tensor(sample[0], dtype=torch.float).to(device)
            y = torch.tensor(sample[1], dtype=torch.float).to(device)
            if args['notes_aggeregate'] == 'Mean' or args['notes_aggeregate'] == 'TimeAttn' or args['notes_aggeregate'] == 'Attn':
                text = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[2]]
                attn = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[3]]
                times = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[4]]
            else:
                text = torch.tensor(sample[2], dtype=torch.long).to(device)
                attn = torch.tensor(sample[3], dtype=torch.long).to(device)
                times = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[4]]
            if sample[2].shape[0] == 0:
                continue
            # Run the model
            Logits, Probs = model(X, text, attn, times)
            # if idx == 0:
            #     with open(os.path.join('Logs', args['checkpoint_path'] + '_ManLog.txt'), 'a+') as f:
            #         print("Text Embedding : ", text_embeddings.detach().cpu(), file=f)
            Lambd = torch.tensor(0.01).to(device)
            l2_reg = model.get_l2()
            loss = model.criterion(Logits, y)
            loss += Lambd * l2_reg
            epoch_loss += loss.item() * y.size(0)
            predicted = Probs.data > 0.5
            if not FirstTime:
                PredScores = Probs
                TrueLabels = y
                PredLabels = predicted
                FirstTime = True
            else:
                PredScores = torch.cat([PredScores, Probs])
                TrueLabels = torch.cat([TrueLabels, y])
                PredLabels = torch.cat([PredLabels, predicted])
            aucpr_obj.add(Probs.detach().cpu(), y.detach().cpu())
        if args['notes_aggeregate'] != 'Flat':
            prf_test = Evaluate(TrueLabels.detach().cpu(), PredLabels.detach().cpu(), PredScores.detach().cpu())
        else:
            prf_test = Evaluate_voting(TrueLabels.detach().cpu(), PredLabels.detach().cpu(), PredScores.detach().cpu(), names)
        prf_test['epoch_loss'] = epoch_loss / TrueLabels.shape[0]
        prf_test['aucpr'] = aucpr_obj.get()
    return prf_test, PredScores

def train_step(model, batch, epoch, num_epochs, display_step, optimizer):
    """
    Train the given model for 1 epoch
    """
    epoch_loss = 0
    total_step = len(batch)
    model.train()
    FirstTime = False
    # Forward pass
    with torch.autograd.set_detect_anomaly(True):  # Error catcher
        for step, sample in enumerate(batch):
            X = torch.tensor(sample[0], dtype=torch.float).to(device)
            y = torch.tensor(sample[1], dtype=torch.float).to(device)
            if args['notes_aggeregate'] == 'Mean' or args['notes_aggeregate'] == 'TimeAttn' or args['notes_aggeregate'] == 'Attn':
                text = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[2]]
                attn = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[3]]
                times = [torch.tensor(x, dtype=torch.float).to(device) for x in sample[4]]
            else:
                text = torch.tensor(sample[2], dtype=torch.long).to(device)
                attn = torch.tensor(sample[3], dtype=torch.long).to(device)
                times = [torch.tensor(x, dtype=torch.float).to(device) for x in sample[4]]
                
            # with open(os.path.join('Logs', args['checkpoint_path'] + '_ManLog.txt'), 'a+') as f:
            #     print('len(text), text, X shape', len(text), text[0].shape, X.shape, file=f)
            # print('len(text), text, X shape', len(text), text[0].shape, X.shape)
            Logits, Probs = model(X, text, attn, times)
            
            Lambd = torch.tensor(0.01).to(device)
            l2_reg = model.get_l2()
            # print('Logits, y shape', Logits.shape, y.shape)
            # with open(os.path.join('Logs', args['checkpoint_path'] + '_ManLog.txt'), 'a+') as f:
            #     print('Logits, y shape', Logits.shape, y.shape, file=f)
            loss = model.criterion(Logits, y)
            loss += Lambd * l2_reg
            with torch.no_grad():
                predicted = Probs.data > 0.5
                if not FirstTime:
                    PredScores = Probs
                    TrueLabels = y
                    PredLabels = predicted
                    FirstTime = True
                else:
                    PredScores = torch.cat([PredScores, Probs])
                    TrueLabels = torch.cat([TrueLabels, y])
                    PredLabels = torch.cat([PredLabels, predicted])
                epoch_loss += loss.item() * y.size(0)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % display_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))
            # if (step+1) % 2 == 0:
            #     break
    with torch.no_grad():
        prf_train = Evaluate(TrueLabels.detach().cpu(), PredLabels.detach().cpu(), PredScores.detach().cpu())
        prf_train['epoch_loss'] = epoch_loss / TrueLabels.shape[0]
    return prf_train

def tokenizeGetIDs(tokenizer, text_data, max_len):
    # Tokenize the texts using tokenizer also pad to max_len and add <cls> token to first and <sep> token to end
    input_ids = []
    attention_masks = []
    if args['notes_aggeregate'] == 'Mean' or args['notes_aggeregate'] == 'TimeAttn' or args['notes_aggeregate'] == 'Attn':
        for texts in text_data:
            if args['NoteSplit'] != 'concat_split':
                Textarr = []
                Attnarr = []
                for text in texts:
                    if args['NoteSplit'] == 'truncate':
                        encoded_sent = tokenizer.encode_plus(
                                text=text,                      # Preprocess sentence
                                add_special_tokens=True,        # Add [CLS] and [SEP]
                                max_length=max_len,             # Max length to truncate/pad
                                pad_to_max_length=True,         # Pad sentence to max length
                                #return_tensors='pt',           # Return PyTorch tensor
                                return_attention_mask=True,     # Return attention mask
                                truncation=True
                                )
                        Textarr.append(encoded_sent.get('input_ids'))
                        Attnarr.append(encoded_sent.get('attention_mask'))
                    elif args['NoteSplit'] == 'note_split':
                         tokenized_txt = text.split()
                         for i in range(0, len(tokenized_txt), max_len-2):
                             encoded_sent = tokenizer.encode_plus(
                                    text=tokenized_txt[i:i+max_len-2],                      # Preprocess sentence
                                    add_special_tokens=True,        # Add [CLS] and [SEP]
                                    pad_to_max_length=True,         # Pad sentence to max length
                                    #return_tensors='pt',           # Return PyTorch tensor
                                    return_attention_mask=True,     # Return attention mask
                                    truncation=True,
                                    max_length=max_len,
                                    is_split_into_words=True
                                    )
                             Textarr.append(encoded_sent.get('input_ids'))
                             Attnarr.append(encoded_sent.get('attention_mask'))
            
                # Add the outputs to the lists
                # input_ids.append(encoded_sent.get('input_ids'))
                # attention_masks.append(encoded_sent.get('attention_mask'))
                input_ids.append(Textarr)
                attention_masks.append(Attnarr)
            else:
                Textarr = []
                Attnarr = []
                tokenized_txt = ' '.join(texts).split()
                for i in range(0, len(tokenized_txt), max_len-2):
                    encoded_sent = tokenizer.encode_plus(
                                            text=tokenized_txt[i:i+max_len-2],                      # Preprocess sentence
                                            add_special_tokens=True,        # Add [CLS] and [SEP]
                                            pad_to_max_length=True,         # Pad sentence to max length
                                            #return_tensors='pt',           # Return PyTorch tensor
                                            return_attention_mask=True,     # Return attention mask
                                            truncation=True,
                                            max_length=max_len,
                                            is_split_into_words=True
                                            )
                    Textarr.append(encoded_sent.get('input_ids'))
                    Attnarr.append(encoded_sent.get('attention_mask'))
                # Add the outputs to the lists
                # input_ids.append(encoded_sent.get('input_ids'))
                # attention_masks.append(encoded_sent.get('attention_mask'))
                input_ids.append(Textarr)
                attention_masks.append(Attnarr)
    else:
        encoded = tokenizer(text_data, max_length=max_len, padding=True, truncation=True)
        input_ids = encoded['input_ids']
        attention_masks = encoded['attention_mask']
    return input_ids, attention_masks

def concat_text_timeseries(data_reader, data_raw):
    # if args['notes_aggeregate'] == 'Concat':
    #     # Read the notes of patient and concatenate the texts together
    #     train_text, train_times, start_time = data_reader.read_all_text_concat_json(data_raw['names'], 48)
    # else:
        # Read the notes of patient and return them as list
    train_text, train_times, start_time = data_reader.read_all_text_append_json(data_raw['names'], 48, NumOfNotes=args['NumOfNotes'], notes_aggeregate = args['notes_aggeregate'])
    
    if args['notes_aggeregate'] == 'Flat':
        # Merge the text data with time-series data but include all the notes as separate data entries
        data = utils.get_text_sep(train_text, data_raw, train_times, start_time)
    else:
        # Merge the text data with time-series data        
        data = utils.merge_text_raw(train_text, data_raw, train_times, start_time)
    return data

def get_time_to_end_diffs(times, starttimes):
    timetoends = []
    for times, st in zip(times, starttimes):
        difftimes = []
        et = np.datetime64(st) + np.timedelta64(49, 'h')
        if args['notes_aggeregate'] == 'Flat':
            time = np.datetime64(times)
            dt = utils.diff_float(time, et)
            assert dt >= 0
            difftimes = dt
        else:
            for t in times:
                time = np.datetime64(t)
                dt = utils.diff_float(time, et)
                assert dt >= 0 #delta t should be positive
                difftimes.append(dt)
        timetoends.append(difftimes)
    return timetoends

def Read_Aggregate_data(mode, AggeragetNotesStrategies, discretizer=None, normalizer = None):
    # mode is between train, test, val
    # Build readers, discretizers, normalizers
    # if os.path.isfile('Data/Train_data_' + AggeragetNotesStrategies + '.pkl'):
    if AggeragetNotesStrategies == 'Flat':
        # There is no difference in data processing between TimeAttn and mean strategies
        File_AggeragetNotesStrategies = 'Flat'
    else:
        File_AggeragetNotesStrategies = 'Mean'
    
    dataPath = os.path.join('Data', mode + '_data_' +  File_AggeragetNotesStrategies + '.pkl')
    if os.path.isfile(dataPath):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            (data, names, discretizer, normalizer) = pickle.load(f)
    else:
        # If we did not already processed the data we do it here
        ReaderPath = os.path.join(conf.ihm_path, 'train' if (mode == 'train') or mode == 'val' else 'test')
        reader = InHospitalMortalityReader(dataset_dir=ReaderPath,
                                                  listfile=os.path.join(conf.ihm_path, mode + '_listfile.csv'), period_length=48.0)
        
        if normalizer is None:
            discretizer = Discretizer(timestep=float(conf.timestep),
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')
        
            discretizer_header = discretizer.transform(
                reader.read_example(0)["X"])[1].split(',')
            cont_channels = [i for (i, x) in enumerate(
                discretizer_header) if x.find("->") == -1]
            
            # text reader for reading the texts
            if (mode == 'train') or (mode == 'val'):
                text_reader = utils.TextReader(conf.textdata_fixed, conf.starttime_path)
            else:
                text_reader = utils.TextReader(conf.test_textdata_fixed, conf.test_starttime_path)
            
            # choose here which columns to standardize
            normalizer = Normalizer(fields=cont_channels)
            normalizer_state = conf.normalizer_state
            if normalizer_state is None:
                normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(
                    conf.timestep, conf.imputation)
                normalizer_state = os.path.join(
                    os.path.dirname(__file__), normalizer_state)
            normalizer.load_params(normalizer_state)
        
            normalizer = None
            # Load the patient data
        train_raw = ihm_utils.load_data(
            reader, discretizer, normalizer, conf.small_part, return_names=True)
        
        print("Number of train_raw_names: ", len(train_raw['names']))
        
        data = concat_text_timeseries(text_reader, train_raw)
        
        train_names = list(data[3])
        
        with open(dataPath, 'wb') as f:
            # Write the processed data to pickle file so it is faster to just read later
            pickle.dump((data, train_names, discretizer, normalizer), f)
        
    data_X = data[0]
    data_y = data[1]
    data_text = data[2]
    data_names = data[3]
    data_times = data[4]
    start_times = data[5]
    timetoends = get_time_to_end_diffs(data_times, start_times)
    
    if args['model_name'] == 'BioBert':
        tokenized_path = os.path.join('Data', EmbedModelName + '_tokenized_ids_attns_' + mode + '_' + File_AggeragetNotesStrategies + '_' + str(args['MaxLen']) + '_' + args['NoteSplit'] + '.pkl')
        if os.path.isfile(tokenized_path):
            # If the pickle file containing text_ids exists we will just load it and save time by not computing it using the tokenizer
            with open(tokenized_path, 'rb') as f:
                txt_ids, attention_masks = pickle.load(f)
            print('txt_ids, attention_masks are loaded from ', tokenized_path)
        else:
            txt_ids, attention_masks = tokenizeGetIDs(tokenizer, data_text, args['MaxLen'])
            with open(tokenized_path, 'wb') as f:
                # Write the output of tokenizer to a pickle file so we can use it later
                pickle.dump((txt_ids, attention_masks), f)
            print('txt_ids, attention_masks is written to ', tokenized_path)
    else:
        txt_ids = data_text.copy()
        attention_masks = list(np.ones_like(txt_ids))
    # Remove the data when text is empty
    indices = []
    for idx, txt_id in enumerate(txt_ids):
        if len(txt_id) == 0:
            indices.append(idx)
        else:
            if args['NumOfNotes'] > 0:
                if args['notes_order'] == 'Last':
                    # Only pick the last note
                    txt_ids[idx] = txt_id[-args['NumOfNotes']:]
                    if args['model_name'] == 'BioBert':
                        attention_masks[idx] = attention_masks[idx][-args['NumOfNotes']:]
                elif args['notes_order'] == 'First':
                    # Only pick the first note
                    txt_ids[idx] = txt_id[:args['NumOfNotes']]
                    if args['model_name'] == 'BioBert':
                        attention_masks[idx] = attention_masks[idx][:args['NumOfNotes']]
            if args['notes_aggeregate'] == 'Concat':
                txt_ids[idx] = ' '.join(txt_ids[idx])
                
    for idx in reversed(indices):
        txt_ids.pop(idx)
        attention_masks.pop(idx)
        data_X = np.delete(data_X, idx, 0)
        data_y = np.delete(data_y, idx, 0)
        data_text = np.delete(data_text, idx, 0)
        data_names = np.delete(data_names, idx, 0)
        data_times = np.delete(data_times, idx, 0)
        start_times = np.delete(start_times, idx, 0)
        timetoends = np.delete(timetoends, idx, 0)
    del data
    
    return txt_ids, attention_masks, data_X, data_y, data_text, data_names, timetoends, discretizer, normalizer

AggeragetNotesStrategies = args['notes_aggeregate']

txt_ids, attention_masks, data_X, data_y, data_text, data_names, data_times, discretizer, normalizer = \
    Read_Aggregate_data('train', AggeragetNotesStrategies, discretizer=None, normalizer = None)

txt_ids_eval, attention_masks_eval, eval_data_X, eval_data_y, eval_data_text, eval_data_names, eval_data_times, _, _ = \
    Read_Aggregate_data('val', AggeragetNotesStrategies, discretizer=discretizer, normalizer = normalizer)

txt_ids_test, attention_masks_test, test_data_X, test_data_y, test_data_text, test_data_names, test_data_times, _, _ = \
    Read_Aggregate_data('test', AggeragetNotesStrategies, discretizer=discretizer, normalizer = normalizer)
    
def generate_tensor_text(t, w2i_lookup, MaxLen):
    # Tokenize for Clinical notes model
    t_new = []
    max_len = -1
    for text in t:
        tokens = list(map(lambda x: lookup(w2i_lookup, x), str(text).split()))
        if MaxLen > 0:
            tokens = tokens[:MaxLen]
        t_new.append(tokens)
        max_len = max(max_len, len(tokens))
    pad_token = w2i_lookup['<pad>']
    for i in range(len(t_new)):
        if len(t_new[i]) < max_len:
            t_new[i] += [pad_token] * (max_len - len(t_new[i]))
    return np.array(t_new)


def generate_padded_batches(x, y, t, text_ids, data_attn, data_times, batch_size, w2i_lookup):
    # Generate batches
    batches = []
    begin = 0
    while begin < len(t):            
        end = min(begin+batch_size, len(t))
        x_slice = np.stack(x[begin:end])
        y_slice = np.stack(y[begin:end])
        if model_name == 'BioBert':
            t_slice = np.array(text_ids[begin:end])
        else:
            t_slice = generate_tensor_text(text_ids[begin:end], w2i_lookup, args['MaxLen'])
        attn_slice = np.array(data_attn[begin:end])
        time_slice = np.array(data_times[begin:end])
        batches.append((x_slice, y_slice, t_slice, attn_slice, time_slice))
        begin += batch_size
    return batches


def validate(model, data_X_val, data_y_val, data_text_val, txt_ids_eval, attn_val, times_val, names_eval, batch_size, word2index_lookup,
             last_best_val_aucpr, save):
    val_batches = generate_padded_batches(
        data_X_val, data_y_val, data_text_val, txt_ids_eval, attn_val, times_val, batch_size, word2index_lookup)
    
    prf_val, probablities = Evaluate_Model(model, val_batches, names_eval)
    loss_value = prf_val['epoch_loss']
        
    final_aucroc = prf_val['roc_macro']
    final_aucpr = prf_val['aucpr']
    print("Validation Loss: %f - AUCPR: %f - AUCROC: %f" %(loss_value, final_aucpr, final_aucroc))

    changed = False
    if final_aucpr > last_best_val_aucpr:
        changed = True
        if save:
            save_path = os.path.join('Checkpoints', args['checkpoint_path'])
            torch.save({'state_dict': model.state_dict()}, save_path)
            print("Best Model saved in", save_path)
    return max(last_best_val_aucpr, final_aucpr), changed, probablities, final_aucroc

def write_probs(PatientNames, test_data_text, probs, test_data_y, path):
    df = pd.DataFrame({'names': PatientNames,
                   'text': test_data_text,
                   'probs': probs.detach().cpu(),
                   'Label': test_data_y})
    df.to_csv(path,index=False)

last_best_val_aucpr = -1
model = ClinicalNotesModel(dropout_keep_prob, W_emb, model_name, model_type, BioBert, args['notes_aggeregate'], args['TSModel'], args['TS_aggeregate'], args['AttnType'], device)
model = model.to(device)
if bool(int(args['load_model'])):
    checkpoint = torch.load(os.path.join('Checkpoints', args['checkpoint_path']))
    model.load_state_dict(checkpoint['state_dict'])
    last_best_val_aucpr, _, probs, val_auc = validate(model, eval_data_X, eval_data_y, eval_data_text, txt_ids_eval, attention_masks_eval, eval_data_times, 
                                                      eval_data_names, batch_size, word2index_lookup, last_best_val_aucpr, False)
    print('Model loaded')

if args['mode'] == 'eval':
    assert bool(int(args['load_model']))
    print('Just Evaluating Mode.')
    last_best_val_aucpr, _, probs, val_auc = validate(model, eval_data_X, eval_data_y, eval_data_text, txt_ids_eval, attention_masks_eval, eval_data_times,
                                                      eval_data_names, batch_size, word2index_lookup, last_best_val_aucpr, False)
    write_probs(eval_data_names, eval_data_text, probs, eval_data_y, "Outputs/Out_probabilities/" + ModelNameOut + "_val.csv")
    
    with open(os.path.join('Outputs', ModelNameOut + '.txt'), 'a+') as f:
        print(" Val AUC : ", val_auc, file=f)
        
    sys.exit(0)

if args['mode'] == 'test':
    assert bool(int(args['load_model']))
    print('Testing Mode.')
    
    last_best_val_aucpr, _, probs, val_auc = validate(model, test_data_X, test_data_y, test_data_text, txt_ids_test, attention_masks_test, test_data_times,
                                                      test_data_names, batch_size, word2index_lookup, last_best_val_aucpr, False)
    write_probs(test_data_names, test_data_text, probs, test_data_y, "Outputs/Out_probabilities/" + ModelNameOut + ".csv")
    
    with open(os.path.join('Outputs', ModelNameOut + '.txt'), 'a+') as f:
        print(" Test AUC : ", val_auc, file=f)
        
    sys.exit(0)

early_stopping = 0
optimizer = torch.optim.Adam(model.parameters(), lr=args['LR'])
print("--- Loaded everything %s seconds ---" % (time.time() - start_time))
with open(os.path.join('Logs', 'ManLogs', args['checkpoint_path'] + '_ManLog.txt'), 'a+') as f:
    print("--- Loaded everything %s seconds ---" % (time.time() - start_time), file=f)
for epoch in range(number_epoch):
    print("Started training for epoch: %d" % epoch)
    data = list(zip(data_X, data_y, data_text, txt_ids, attention_masks, data_times))
    random.shuffle(data)
    data_X, data_y, data_text, txt_ids, attention_masks, data_times = zip(*data)

    del data

    batches = generate_padded_batches(
        data_X, data_y, data_text, txt_ids, attention_masks, data_times, batch_size, word2index_lookup)

    print("Generated batches for the epoch!")

    prf_train = train_step(model, batches, epoch, number_epoch, 500, optimizer)
    aucroc_value = prf_train['roc_macro']
    loss_value = prf_train['epoch_loss']

    current_aucroc = aucroc_value
    print("Loss: %f - AUCROC: %f" %(loss_value, current_aucroc))

    loss_list = []

    del batches
    print("Started Evaluation After Epoch : %d" % epoch)
    last_best_val_aucpr, changed, probs, val_auc = validate(model, eval_data_X, eval_data_y, eval_data_text, txt_ids_eval, attention_masks_eval, eval_data_times,
                                                            eval_data_names, batch_size, word2index_lookup, last_best_val_aucpr, True)
    with open(os.path.join('Logs', 'ManLogs', args['checkpoint_path'] + '_ManLog.txt'), 'a+') as f:
        print("Epoch : ", epoch, " time : ", (time.time() - start_time), " Val AUC : ", val_auc, file=f)
    if changed == False:
        early_stopping += 1
        print("Didn't improve!: " + str(early_stopping))
    else:
        early_stopping = 0

    if early_stopping >= 15:
        print(
            "AUCPR didn't change from last 15 epochs, early stopping")
        break
    print("*End of Epoch.*\n")