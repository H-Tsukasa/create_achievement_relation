# t5_purpose_abstraction
## 実行
### モデルの学習
ディレクトリを移動．
```
cd ./src/t5_purpose_abstraction
```
学習を実行．
```
python3 train.py
```
### モデルの検証
use_dataフォルダに利用したいデータを追加．(csv, txt)  

検証を実行（ファイル名が無い場合は purposes.txt で検証）
```
python3 test.py ファイル名 
```