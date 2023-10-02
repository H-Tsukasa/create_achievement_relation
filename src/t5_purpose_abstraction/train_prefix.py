from pathlib import Path
import re
import math
import time
import demoji
import mojimoji
import copy
import glob
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.model_selection import train_test_split

MODEL_NAME = "sonoisa/t5-base-japanese"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, is_fast=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length_src = 400
max_length_target = 200
batch_size_train = 8
batch_size_valid = 8
epochs = 1000
patience = 10

def text_revise(text):
    text = text.replace('\n','').replace('\r','') #改行削除
    text = text.replace(' ', '') #スペース削除
    text = text.replace('　', '')
    text = demoji.replace(string=text, repl='') #絵文字削除
    text = re.sub(r'[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]', '', text)
    text = re.sub("[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", '', text)
    text = re.sub(r'\b\d{1,3}(,\d{3})*\b', '0', text) #0に変換
    text = re.sub(r'\d+', '0', text) #0に変換
    text = text.lower() #大文字を小文字に変換
    text = mojimoji.han_to_zen(text) #半角から全角
    return text
  
def convert_batch_data(train_data, valid_data, tokenizer):

    def generate_batch(data):

        batch_src, batch_tgt = [], []
        for src, tgt in data:
            batch_src.append(src)
            batch_tgt.append(tgt)

        batch_src = tokenizer(
            batch_src, max_length=max_length_src, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_tgt = tokenizer(
            batch_tgt, max_length=max_length_target, truncation=True, padding="max_length", return_tensors="pt"
        )

        return batch_src, batch_tgt

    train_iter = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(valid_data, batch_size=batch_size_valid, shuffle=True, collate_fn=generate_batch)

    return train_iter, valid_iter
  
class T5FineTuner(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None,
        decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
          )       
        
def train(model, data, optimizer, PAD_IDX):

  model.train()

  loop = 1
  losses = 0
  pbar = tqdm(data)
  for src, tgt in pbar:

      optimizer.zero_grad()

      labels = tgt['input_ids'].to(device)
      labels[labels[:, :] == PAD_IDX] = -100

      outputs = model(
          input_ids=src['input_ids'].to(device),
          attention_mask=src['attention_mask'].to(device),
          decoder_attention_mask=tgt['attention_mask'].to(device),
          labels=labels
      )
      loss = outputs['loss']

      loss.backward()
      optimizer.step()
      losses += loss.item()

      pbar.set_postfix(loss=losses / loop)
      loop += 1

  return losses / len(data)

def evaluate(model, data, PAD_IDX):

    model.eval()
    losses = 0
    with torch.no_grad():
        for src, tgt in data:

            labels = tgt['input_ids'].to(device)
            labels[labels[:, :] == PAD_IDX] = -100

            outputs = model(
                input_ids=src['input_ids'].to(device),
                attention_mask=src['attention_mask'].to(device),
                decoder_attention_mask=tgt['attention_mask'].to(device),
                labels=labels
            )
            loss = outputs['loss']
            losses += loss.item()

    return losses / len(data)





###前処理###
file = glob.glob('../../datas/pos_datas/*.txt')
purpose_texts = []
purposes = []
purpose_pairs = []
for f in file:
  with open(f, "r") as f:
    for i, line in enumerate(f.readlines()):
      if not i == 13 and not i == 27:
        line = line.replace("\n", "")
        split_line = line.split(",")
        for j, s in enumerate(split_line[3:]):
          if not s == "" and j%2==1 and not split_line[3+j-1] == "":
              purpose_texts.append(split_line[3+j-1])
              purposes.append(s)
              purpose_pairs.append((split_line[3+j-1], s))
              
purposes = [text_revise(purpose) for purpose in purposes]
purposes = ["要約:「"+purpose+"」の目的は" for purpose in purposes]
purpose_texts = [text_revise(purpose_text) for purpose_text in purpose_texts]
use_datas = list(zip(purposes, purpose_texts))

X_train, X_test, y_train, y_test = train_test_split(purpose_texts, purposes, test_size=0.2, random_state=1)
train_data = [(src, tgt) for src, tgt in zip(X_train, y_train)]
valid_data = [(src, tgt) for src, tgt in zip(X_test, y_test)]

train_iter, valid_iter = convert_batch_data(train_data, valid_data, tokenizer)




### 学習 ###
model = T5FineTuner()
model = model.to(device)

optimizer = optim.Adam(model.parameters())

PAD_IDX = tokenizer.pad_token_id
best_loss = float('Inf')
best_model = None
counter = 1

for loop in range(1, epochs + 1):
    start_time = time.time()
    loss_train = train(model=model, data=train_iter, optimizer=optimizer, PAD_IDX=PAD_IDX)
    elapsed_time = time.time() - start_time
    loss_valid = evaluate(model=model, data=valid_iter, PAD_IDX=PAD_IDX)
    print('[{}/{}] train loss: {:.4f}, valid loss: {:.4f} [{}{:.0f}s] counter: {} {}'.format(
        loop, epochs, loss_train, loss_valid,
        str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
        elapsed_time % 60,
        counter,
        '**' if best_loss > loss_valid else ''
    ))
    if best_loss > loss_valid:
        best_loss = loss_valid
        best_model = copy.deepcopy(model)
        counter = 1
    else:
        if counter > patience:
            break
        counter += 1

###モデルの保存
path = './model_prefix'
if not os.path.exists(path):
    os.makedirs(path)

model_dir_path = Path('model_prefix')
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)

tokenizer.save_pretrained(model_dir_path)
best_model.model.save_pretrained(model_dir_path)

tokenizer = T5Tokenizer.from_pretrained(model_dir_path)
trained_model = T5ForConditionalGeneration.from_pretrained(model_dir_path)
trained_model = trained_model.to(device)