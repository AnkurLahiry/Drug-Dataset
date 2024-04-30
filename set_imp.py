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

from datasets import Dataset as D

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

from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitTrainer, Trainer, SetFitModel, TrainingArguments

def cleanup():
    dist.destroy_process_group()

spacy.require_gpu()

nlp = spacy.load("en_core_web_sm",disable=["parser", "ner"])

train_data_path = "dataset/drugsComTrain_raw.csv"
test_data_path = "dataset/drugsComTest_raw.csv"

from torch.utils.data import Dataset, DataLoader, DistributedSampler

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f'Is GPU initialized on {rank}: {dist.is_initialized()}')

def cleanup():
    dist.destroy_process_group()
    
def convert_hugging_dataset(X, y):
    data_dicts = [{'text': input_data, 'label': target_data} for input_data, target_data in zip(X, y)]
    print(len(data_dicts))
    value = D.from_dict({key: [dic[key] for dic in data_dicts] for key in data_dicts[0]})
    print(value)
    return D.from_dict({key: [dic[key] for dic in data_dicts] for key in data_dicts[0]})


def implement_sentence_fit(X_train, y_train, X_val, y_val, X_test, y_test, device):
    
    train_dataset = convert_hugging_dataset(X_train, y_train)
    validation_dataset = convert_hugging_dataset(X_val, y_val)
    test_dataset = convert_hugging_dataset(X_test, y_test)
    
    model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
    model = SetFitModel.from_pretrained(model_id, multi_target_strategy="one-vs-rest")
    model = model.to(device)
    
    args = TrainingArguments(
       batch_size=1024,
       num_epochs=4,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
    )
    
    trainer = Trainer(
    	model=model,
    	args=args,
    	train_dataset=train_dataset,
    	eval_dataset=validation_dataset,
    	metric="accuracy",
    	column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
    )
    
    trainer.train()

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
    X_train, y_train = train_data['processed_text'], train_data['rating'].astype(float)
    X_test, y_test = test_data['processed_text'], test_data['rating'].astype(float)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    implement_sentence_fit(X_train, y_train, X_val, y_val, X_test, y_test, device)

def fn(num_gpus):
    mp.spawn(run, args=(num_gpus), nprocs=num_gpus)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, ), nprocs=world_size) #fn(world_size)
