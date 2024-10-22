import faulthandler
faulthandler.enable()
from Bio import SeqIO
import numpy as np
import argparse
import csv
import re
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd
from collections import namedtuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from evo import Evo
from transformers import AutoTokenizer
from tqdm import tqdm
from kmer import kmer_featurization
import torch.nn.functional as F
import torch
import math
import torch.nn as nn


parser.add_argument('--input', type=str, help='name of the input file (fasta format)')
parser.add_argument('--output', type=str, help='output directory', default='result')
parser.add_argument('--len', type=int, help='predict only for sequences >= len bp (default: 500)', default=500)
parser.add_argument('--threshold', type=float, help='threshold for prediction (default: 0.5)', default=0.5)
inputs = parser.parse_args()

input_pth = inputs.input
output_path = inputs.output
batch_size = inputs.batch_size
len_threshold = int(inputs.len)
score_threshold = float(inputs.threshold)
cpu_threads = int(inputs.threads)
model_pth = 'model'
filename = input_pth.rsplit('/')[-1].split('.')[0]

if score_threshold < 0.5:
    print('Error: Threshold for prediction must be >= 0.5')
    exit(1)

if output_path == '':
    print('Error: Please specify a directory for output')
    exit(1)

if not os.path.isdir(output_path):
    os.makedirs(output_path)


def special_match(strg, search=re.compile(r'[^ACGT]').search):
    return not bool(search(strg))


def preprocee_data(input_pth, output_path, len_threshold):
    frag_len = 1024
    filename = input_pth.rsplit('/')[-1].split('.')[0]
    f = open(f"{output_path}/{filename}_temp.csv", "w")
    f.write(f'sequences,ids\n')
    copy_nums = []
    for record in SeqIO.parse(input_pth, "fasta"):
        sequence = str(record.seq).upper()
        if special_match(sequence):
            copy_num = 0
            last_pos = 0
            if len(sequence) < len_threshold:
                continue
            else:   
               for i in range(0, len(sequence)-frag_len+1, 1024):
                    if len(sequence) - last_pos < 2048:
                        f.write(f'{sequence[last_pos:]},{f"{record.id}_{last_pos - 0}_{len(record.seq)}"}\n')
                        copy_num += 1
                        continue
                    sequence1 = sequence[i:i + frag_len]
                    if special_match(sequence1):
                        f.write(f'{sequence1},{f"{record.id}_{i}_{i+frag_len}"}\n')
                        copy_num += 1
                    last_pos = i+frag_len
            copy_nums.append(copy_num)
    f.close()
    #dataframe = pd.DataFrame({'ids':ids, 'sequences':sequences})
    #dataframe.to_csv(output_path +'/orginal.csv', index=False, sep=',')
    return copy_nums

class embedding_Dataset(Dataset):
    def __init__(self, path):
        self.sequences = pd.read_csv(path)['sequences']
        print('the amount of contigs:', len(self.sequences))
        self.sequences = [sequence for sequence in self.sequences]#[:1000000]
        self.sequences = preprocess_function(self.sequences)
        self.input_ids = self.sequences['input_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_id = self.input_ids[index]
        return input_id

class myDataset(Dataset):
    def __init__(self, emb_path, kmer_path):
        self.emb = torch.load(emb_path) 
        print('test contigs:', len(self.emb))
        self.kmer = np.load(kmer_path)
        self.labels = [1 for i in range(40000000)] 
        print(len(self.emb))
    
    def __len__(self):
        return len(self.emb)
    
    def __getitem__(self, index):
        emb = self.emb[index].view(-1)
        kmer = self.kmer[index]
        label = self.labels[index]
        return (emb, kmer, label)

class CustomEmbedding(nn.Module):
  def unembed(self, u):
    return u

def print_performance(epoch, labels, preds):
    print('Epoch', epoch)
    print('accuracy', accuracy_score(labels, preds))
    print('precision', precision_score(labels, preds, average='macro'))
    print('recall', recall_score(labels, preds, average='macro'))
    print('F1-score', f1_score(labels, preds, average='macro'))

def test(model, device, test_loader, epoch, loss_fn):
    """Test loop."""
    model.eval()
    test_loss = 0
    correct = 0
    result, label, outputs, scores = [], [], [], []
    with torch.no_grad():
        for (emb, kmer, target) in test_loader:
            emb, kmer, target = emb.to(device), kmer.type(torch.float32).to(device), target.to(device)
            output = model(emb, kmer)
            outputs = np.append(outputs, output[:,0].cpu())
            test_loss += loss_fn(output, target).sum().item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            result = np.append(result, pred.cpu())
            #print(output.shape)
            score = F.softmax(output, dim=1)[:,1]
            scores = np.append(scores, score)
            label = np.append(label, target.view_as(pred).cpu())
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print('labels', label[:100])
    print('preds', result[:100])
    #plot_roc(label, outputs)
    #print_performance(epoch, label, result)
    return test_loss, scores

def preprocess_function(sample):
    model_name = 'togethercomputer/evo-1-8k-base'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "X"
    return tokenizer(sample, padding="longest", truncation=True, max_length=1024)

def inference(model, input_ids):
    device = 'cuda'
    input_ids = torch.stack(input_ids)
    input_ids = torch.transpose(input_ids, 0, 1)
    input_ids = input_ids.to(device)
    with torch.inference_mode():
        logits, _ = model(input_ids) # (batch, length, vocab)
    logits = logits.to('cpu')
    logits = torch.mean(logits, dim=1).float()
    input_ids = input_ids.to('cpu')
    return logits

def restore_embedding(path, embedding_path):
    device = 'cuda'
    amount = 0
    batch_size = 32
    ds = embedding_Dataset(path)
    print('the numer of split contigs:', len(ds))
    embedding_list = []
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    evo_model = Evo('evo-1-8k-base')
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()
    model.unembed = CustomEmbedding()
    for sequences in tqdm(data_loader):
        #print('{}/{}'.format(amount * batch_size, len(ds)))
        amount += 1
        emb = inference(model, sequences)
        embedding_list.extend(torch.split(emb, 1, dim=0))
    torch.save(embedding_list, embedding_path)
    torch.cuda.empty_cache()

def restore_kmer(path, kmer_path):
    sequences = pd.read_csv(path, encoding='utf-8')['sequences']
    print(len(sequences))
    obj = kmer_featurization(5)
    kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(sequences, write_number_of_occurrences=False)

    np.save(kmer_path, np.array(kmer_features, dtype=np.float32))

path = output_path + '/' + filename + '_temp.csv'
embedding_path = output_path + '/embedding.pt'
if not os.path.exists(path):
    print('---splitting starts---')
    copy_nums = preprocee_data(input_pth, output_path, len_threshold)
else:
    print('split csv exists')

if not os.path.exists(embedding_path):
    print('---embedding starts---')
    restore_embedding(path, embedding_path)
else:
    print('embedding exists')

og_path = output_path+'/orginal.csv'
kmer_path = output_path + '/kmer.npy'
if not os.path.exists(kmer_path):
    print('---kmer module starts---')
    restore_kmer(path, kmer_path)
else:
    print('kmer exists')


#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
batch_size = 64
epoch = 1
model = torch.load('weight.pth')
model.to(device)
ds = myDataset(embedding_path, kmer_path)
data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
loss_fn = nn.CrossEntropyLoss()
test_loss, scores = test(model, device, data_loader, epoch, loss_fn)
ids = pd.read_csv(path)['ids']
dataframe = pd.DataFrame({'ids':ids, 'scores':scores})
dataframe.to_csv(output_path+'/scores.csv', index=False, sep=',')


