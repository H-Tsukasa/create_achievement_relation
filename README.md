# create_achievement_relation
このレポジトリでは，言語モデルを用いて目的を抽出，抽象化するコードと，アチーブメント関係を抽出する手法をまとめています．
## 開発環境
#### バージョン
- Ubuntu 22.04.3 LTS
- python 3.9.1
- cuda 11.7
#### 主なライブラリ
```
transformers==4.33.1
scikit-learn==1.2.1
unidic_lite==1.0.8
torch==1.13.0
torchvision==0.14.0 
torchaudio==0.13.0
```
## 構成
```
├── datas
│   ├── neg_datas #正例データ
│   └── pos_datas #負例データ
├── pyproject.toml
├── README.md
├── requirements.txt
└── src
    ├── bert_purpose_classification
    │   ├── lib
    │   │   ├── data_processing.py
    │   │   ├── EarlyStopping.py
    │   │   ├── evaluation.py
    │   │   ├── fine_tuning.py
    │   │   ├── Input.py
    │   │   └── preprocess.py
    │   ├── test.py #学習済みBERTの検証
    │   ├── test_results
    │   │   ├── non_purpose.txt
    │   │   └── purpose.txt
    │   ├── train.py #BERTモデル学習
    │   └── use_data
    │       ├── sentences.csv
    │       ├── test.csv
    │       ├── train.csv
    │       └── val.csv
    └── t5_purpose_abstraction
        ├── test.py #学習済みt5の検証
        ├── test_results
        │   └── result.txt
        ├── train.py #t5の学習
        └── use_data
            └── purposes.txt
    └── create_acheivement
        ├── convert_vec.py
        ├── grouping_by_sbert.py
        ├── purpose_abstraction.py
        └── purpose_extraction.py
```
- datas  
学習に利用する目的を含む正例データ，目的を含まない負例データを格納
- bert_purpose_classification  
文章が目的を含むかどうか分類するBERTを学習するフォルダ
- t5_purpose_abstraction  
目的を含む文章を入力すると，抽象化された目的を返すt5を学習するフォルダ．
## 準備
requirements.txtのライブラリをインストール．
```
pip install -r requirements.txt
```
## 実行
各フォルダに使い方を掲載．