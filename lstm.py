import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd # Data Processing, CSV file I/O
import os # This Module Provides a Portable Way of Using Operating System-Dependent Functionality
import random
import numpy as np # Linear Algebra 
np.random.seed(42) #For Reproducibility
import matplotlib.pyplot as plt # Plotting library in Python
from matplotlib import style
import seaborn as sns # Statistical Data Visualization
from scipy import stats # Scientific Computing
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import string
import html
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from spacy.lang.en import English
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import torch.optim as optim
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast  # Required for AMP
import transformers
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.multiprocessing import Process
import torchtext
from sklearn.utils.class_weight import compute_class_weight
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import time
import warnings

from datetime import datetime
#import datetime

#from datetime import utcnow

from time import gmtime, strftime

warnings.filterwarnings("ignore")

torchtext.disable_torchtext_deprecation_warning()

from concurrent.futures import ThreadPoolExecutor

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#nltk.download('vader_lexicon')
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from util import *
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

def cleanup():
    dist.destroy_process_group()

spacy.require_gpu()

nlp = spacy.load("en_core_web_sm",disable=["parser", "ner"])

train_data_path = "dataset/drugsComTrain_raw.csv"
test_data_path = "dataset/drugsComTest_raw.csv"

from torch.utils.data import Dataset, DataLoader, DistributedSampler

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f'Is GPU initialized on {rank}: {dist.is_initialized()}')

def cleanup():
    dist.destroy_process_group()

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0),  # Dropout only between LSTM layers if num_layers > 1
            batch_first=True,
            bidirectional=bidirectional
        )
        # Adjusting the Dimension of the FC Layer if the LSTM is Bidirectional
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # Embedding and applying dropout
        embedded = self.dropout(self.embedding(text))
        # Pack sequence
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # Handling output from bidirectional LSTM
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        # Apply dropout to the output of the LSTM
        hidden = self.dropout(hidden)
        # Fully connected layer to get logits
        logits = self.fc(hidden)
        return logits

def train_and_evaluate(model, train_loader, 
                        val_loader,train_sampler, 
                        validation_sampler, optimizer, 
                        criterion,rank, device, epochs=1):
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_sampler.set_epoch(epoch)
        for inputs, labels, lengths in train_loader:  # Updated to Unpack Lengths
            #device = torch.device("cpu")
            labels = labels - 1
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths).to(device)  # Passing Lengths to the Model
            loss = criterion(outputs.to(device), labels.long())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        validation_sampler.set_epoch(epoch)
        val_loss = []
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:  # Updated to Unpack Lengths
                labels = labels - 1
                inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
                outputs = model(inputs, lengths).to(device)  # Passing Lengths to the Model
                loss = criterion(outputs.to(device), labels.long())
                val_loss.append(loss.item())

        avg_val_loss = np.average(val_loss)
        history['val_loss'].append(avg_val_loss)
        if rank == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        #dist.barrier()    
    return model, avg_train_loss, avg_val_loss, history

class RatingDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.texts[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return text_tensor, label_tensor


def collate_batch(batch, vocab):
    texts, labels = zip(*batch)
    # Checking for empty texts directly in the list of processed indices
    text_tensors = [torch.tensor(x, dtype=torch.long) if len(x) > 0 else torch.tensor([vocab['<pad>']], dtype=torch.long) for x in texts]
    lengths = torch.tensor([len(x) for x in text_tensors], dtype=torch.long)
    texts_padded = pad_sequence(text_tensors, batch_first=True, padding_value=vocab['<pad>'])
    labels = torch.tensor(labels, dtype=torch.float32)
    return texts_padded, labels, lengths

def lstm(X_train, y_train, X_val, y_val, X_test, y_test, vocab, device, rank, world_size):
    print("function called")
    train_dataset = RatingDataset(X_train.values, y_train.values)
    validation_dataset = RatingDataset(X_val.values, y_val.values)
    test_dataset = RatingDataset(X_test.values, y_test.values)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=512, 
                                pin_memory=True, num_workers=0, 
                                drop_last=False, collate_fn=lambda b: collate_batch(b, vocab))
    validation_loader = DataLoader(validation_dataset, batch_size=256, pin_memory=True, 
                                    num_workers=0, drop_last=False, 
                                        collate_fn=lambda b: collate_batch(b, vocab))
    test_loader = DataLoader(test_dataset, batch_size=256, pin_memory=True, 
                                num_workers=0, drop_last=False, 
                                    collate_fn=lambda b: collate_batch(b, vocab))

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    #device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device)).to(device)
    param_grid = {
        'embed_dim': 50,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.5,
        'learning_rate': 0.001
    }
    #optimizer = optim.Adam(model.parameters(), lr=param_grid['learning_rate'])
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=param_grid['embed_dim'],
        hidden_dim=param_grid['hidden_dim'],
        output_dim=len(set(y_train)),
        num_layers=param_grid['num_layers'],
        dropout=param_grid['dropout'],
        bidirectional=True  
    ).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=param_grid['learning_rate'])
    model, train_loss, validation_loss, _ = train_and_evaluate(model, 
                                                    train_loader, validation_loader,
                                                    train_sampler, validation_sampler, 
                                                    optimizer, criterion, 
                                                    rank, device, epochs=1)    


    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels, lengths in test_loader:  # Updated to Unpack Lengths
            labels = labels - 1
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths).to(device)  # Passing Lengths to the Model
            loss = criterion(outputs.to(device), labels.long())
            y_true.extend(labels.cpu().numpy().tolist())
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            y_pred.extend(predicted.cpu().numpy().tolist())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')  # 'weighted' for unbalanced classes
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if rank == 0:
        print(f'Accuracy: {accuracy}')
        print(f'F1 Score: {f1}')
        print(f'Matthews Correlation Coefficient: {mcc}')
        show_confusion_matrix(cm, y_true, y_pred, 'lstm_confusion_matrix.pdf')
        show_train_validation_loss(train_loss, validation_loss, 'lstm_train_loss.pdf')
        
def preprocess_text(text):
    text = html.unescape(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return [token.lemma_ for token in nlp(text.lower()) if not token.is_stop and token.is_alpha]

def preprocess_texts(texts, num_workers=4):
    """Process a list of texts in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(preprocess_text, texts))
    return results

def review_preprocess(df, datapath, rank, devices):
    data = df["review"].values
    world_size = torch.cuda.device_count()
    results = preprocess_texts(data, num_workers=world_size)
    dist.barrier()
    df["processed_text"] = results
    df.to_csv(datapath, index=False)
    return df

def preprocess(proc_id, devices):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    if proc_id == 0:
         print(f'Train Dataset shape: {train_data.shape}')
         print(f'Test Dataset shape: {test_data.shape}')
         print(f'Columns: {train_data.columns}')
    dist.barrier()
    #del train_data['processed_text']
    #del test_data['processed_text']
    utc_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(utc_time)
    if 'processed_text' not in train_data:
        train_data = review_preprocess(train_data, train_data_path, proc_id, devices)
    if 'processed_text' not in test_data:
        test_data = review_preprocess(test_data, test_data_path, proc_id, devices)
    utc_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(utc_time)
    return train_data, test_data

def build_vocab(data):
    counter = Counter()
    for text in data:
        counter.update(text)
    vocab = build_vocab_from_iterator([counter.keys()], specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]

#vocab = build_vocab(train_df['processed_text'])

def get_vocab(df):
    vocab = build_vocab(df['processed_text'])

def run(rank, world_size):
    # Initialize distributed training context.
    setup(rank, world_size) 
    torch.cuda.set_device(rank)
    train_data, test_data = preprocess(rank, world_size)
    if 'processed_indices' not in train_data:
        vocab = build_vocab(train_data['processed_text'])
        train_data['processed_indices'] = train_data['processed_text'].apply(lambda x: tokens_to_indices(x, vocab))
        test_data['processed_indices'] = test_data['processed_text'].apply(lambda x: tokens_to_indices(x, vocab))
    #vocab = get_vocab(train_data)
    X_train, y_train = train_data['processed_indices'], train_data['rating'].astype(float)
    X_test, y_test = test_data['processed_indices'], test_data['rating'].astype(float)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    dist.barrier()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    lstm(X_train, y_train, X_val, y_val, X_test, y_test, vocab, device, rank, world_size)

def fn(num_gpus):
    mp.spawn(run, args=(num_gpus), nprocs=num_gpus)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, ), nprocs=world_size) #fn(world_size)
    #assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    #world_size = n_gpus
    #mp.spawn(run, args=(list(range(num_gpus)),), nprocs=num_gpus) #mp.spawn(run, args=(world_size,), nprocs=world_size, join=True) 
    #cleanup()
