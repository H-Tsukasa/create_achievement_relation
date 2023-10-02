import re
import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import transformers
import re
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from lib.Input import *
from lib.SequenceClassification import SequenceClassification
from lib.fine_tuning import fine_tuning
from lib.evaluation import *
from lib.preprocess import *
from lib.data_processing import *

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

### 利用するファイルの定義 ###
file = glob.glob('../../datas/pos_datas/*.txt')
file2 = glob.glob('../../datas/neg_datas/*.txt')


### 入力データに関する変数 ###
text_column_name = 'texts'
pair_text_column_name = None
label_names = ['目的', '非目的']
label_column_name = 'labels'
max_seq_length = 51

### データの保存に関する変数 ###
save_dir = './'
folder_name = 'use_data'
train_csv_name = 'train.csv'
val_csv_name = 'val.csv'
test_csv_name = 'test.csv'

### データの読み込みに関する変数 ###
load_file_dir = './use_data'
load_train_file_name = 'train.csv'
load_val_file_name = 'val.csv'
# load_test_file_name = 'test.csv'
load_test_file_name = 'sentence.csv'

### 学習に関する変数 ###
pretrained_path = 'cl-tohoku/bert-base-japanese-v3'
tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(pretrained_path)
batch_size = 2

### checkpointに関する変数 ###
save_checkpoint_dir = './checkpoints_v1'
checkpoint_name = 'bert_fine_tuning_checkpoint_for_binary_classification'
checkpoint_path = os.path.join(save_checkpoint_dir, "{}".format(checkpoint_name + '.pt'))


### 準備 ###
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())
#     print('We will use the GPU:', torch.cuda.get_device_name(0))
# else:
#     device = torch.device("cpu")
#     print('No GPU available, using the CPU instead.')


# 前処理
# purpose_texts = []
# purposes = []
# purpose_pairs = []
# for f in file:
#   # print(f)
#   with open(f, "r") as f:
#     for i, line in enumerate(f.readlines()):
#       if not i == 13 and not i == 27:
#         line = line.replace("\n", "")
#         split_line = line.split(",")
#         for j, s in enumerate(split_line[3:]):
#           if not s == "" and j%2==1 and not split_line[3+j-1] == "":
#               purpose_texts.append(split_line[3+j-1])
#               purposes.append(s)
#               purpose_pairs.append((split_line[3+j-1], s))

# # skip_reviewの取得
# skip_reviews = []
# for fname in file2:
#   with open(fname, "r") as f:
#     for line in f.readlines():
#       split_line = line.split(",")
#       split_review_texts = re.split('[．。！？\!\?]', split_line[1].replace("\n",""))
#       for split_review_text in split_review_texts:
#         if not split_review_text == "":
#           skip_reviews.append(split_review_text)

# # 前処理
# purposes = [text_revise(purpose) for purpose in purposes]
# purpose_texts = [text_revise(purpose_text) for purpose_text in purpose_texts]
# skip_reviews = [text_revise(sr) for sr in skip_reviews]

# cor_list =["目的"] * len(purpose_texts)
# incor_list = ["非目的"] * len(skip_reviews)
# len(incor_list)

# df_list = list(zip(purpose_texts, cor_list))
# df_list += list(zip(skip_reviews, incor_list))
# texts = [text[0] for text in df_list]
# labels = [text[1] for text in df_list]


# X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=1, stratify=labels)
# X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1, stratify=y_test)

# これらのリストをpandasのDataFrameに格納
# 訓練データ
# train_df = pd.DataFrame({'texts': X_train,
#                          'labels': y_train})
# 検証データ
# val_df = pd.DataFrame({'texts': X_valid,
                      #  'labels': y_valid})
# テストデータ
# test_df = pd.DataFrame({'texts': X_test,
                        # 'labels': y_test})

# 作業ディレクトリ内にデータを保存するフォルダを新規作成する
# フォルダの作成
# make_folder(save_dir, folder_name)
# Dataframeのindexを揃える
# train_df = train_df.set_index('texts')
# val_df = val_df.set_index('texts')
# test_df = test_df.set_index('texts')
# CSVとして保存
# train_df.to_csv(save_dir + folder_name + '/' + train_csv_name,
#                 encoding = 'utf_8_sig')
# val_df.to_csv(save_dir + folder_name + '/' + val_csv_name,
#               encoding = 'utf_8_sig')
# test_df.to_csv(save_dir + folder_name + '/' + test_csv_name,
#                encoding = 'utf_8_sig')

# 訓練データの読み込み＆変換
# train_tds = load_and_cache_examples(
#         data_dir=load_file_dir, mode=load_train_file_name, max_seq_length=max_seq_length,
#         tokenizer=tokenizer, pad_token_label_id=0, label_names=label_names,
#         text_column_name=text_column_name, pair_text_column_name=pair_text_column_name,
#         label_column_name=label_column_name)

# 検証データ読み込み＆変換
# val_tds = load_and_cache_examples(
#         data_dir=load_file_dir, mode=load_val_file_name, max_seq_length=max_seq_length,
#         tokenizer=tokenizer, pad_token_label_id=0, label_names=label_names,
#         text_column_name=text_column_name, pair_text_column_name=pair_text_column_name,
#         label_column_name=label_column_name)

# テストデータ読み込み＆変換
test_tds = load_and_cache_examples(
        data_dir=load_file_dir, mode=load_test_file_name, max_seq_length=max_seq_length,
        tokenizer=tokenizer, pad_token_label_id=0, label_names=label_names,
        text_column_name=text_column_name, pair_text_column_name=pair_text_column_name,
        label_column_name=label_column_name)

# dataloaderに格納(データをbatch_size毎に整えるため)
# train_dataloader = DataLoader(train_tds, batch_size=batch_size, shuffle=False)
# val_dataloader = DataLoader(val_tds, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_tds, batch_size=batch_size, shuffle=False)

# 訓練データと検証データをdictにまとめる
# dataloaders_dict = {'train_train': train_dataloader,
#                     'train_eval': train_dataloader,
#                     'valid_eval': val_dataloader}

# BERTモデルの定義(num_clabelsはn値分類の場合はn)
model = SequenceClassification.from_pretrained(pretrained_path, num_clabels=2)

#BERTモデル内の各transformer層のパラメータ更新をTrueに
# for name, param in model.named_parameters():
#     param.requires_grad = True

# 分類層（最後の全結合）のパラメータの更新する
# for name, param in model.classifier.named_parameters():
#     param.requires_grad = True
    
# チェックポイントの作成
# make_folder('./', 'checkpoints')

### 訓練開始 ###
# fine_tuning(model=model,
#             dataloaders_dict=dataloaders_dict,
#             learning_rate=2e-5,
#             num_epochs=3,
#             checkpoint_path=checkpoint_path,
#             restart=None,
#             save_epoch_for_interval_model=30,
#             use_vector='cls',
#             loss_weight=None,
#             study_early_stop=True,
#             patience=3,
#             print_batch_log=True,
#             device=device,
#             batch_size=batch_size)

### 検証 ###
# チェックポイントから学習したパラメータを読み込む
model = model_load_checkpoint(model, checkpoint_path)

# 検証の実行
outputs = test_prediction(model=model,
                          input_dataloader=test_dataloader,
                          tokenizer=tokenizer,
                          use_vector='cls',
                          label_names=label_names)


output_df = pd.DataFrame(outputs)

purpose_df = output_df.query('予測ラベル=="目的"')
non_purpose_df = output_df.query('予測ラベル=="非目的"')

purposes = list(purpose_df["入力文"])
non_purposes = list(non_purpose_df["入力文"])

purposes = [p.replace("[CLS]", "").replace("[SEP]", "") for p in purposes]
non_purposes = [p.replace("[CLS]", "").replace("[SEP]", "") for p in non_purposes]


with open("../../datas/bert_result/purpose_v1.txt", "w") as f:
    for p in purposes:
        f.write(p)
        f.write("\n")

with open("../../datas/bert_result/non_purpose_v1.txt", "w") as f:
    for p in non_purposes:
        f.write(p)
        f.write("\n")