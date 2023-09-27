import os
import re
import mojimoji
import demoji
# テキストとラベルを分ける関数
def get_texts_and_labels(tupples):
    text_lis = []
    label_lis = []
    for one_data_tupple in tupples:
        text = one_data_tupple[1]
        label = one_data_tupple[0]
        text_lis.append(text)
        label_lis.append(label)
    return text_lis, label_lis

# ディレクトリ上にフォルダを新規作成する関数
def make_folder(dir, folder_name):
    # 作りたいフォルダのpathを指定
    save_fig_folder_path = os.path.join(dir, "{}".format(folder_name))
    # フォルダの作成
    try:
        os.makedirs(save_fig_folder_path)
    except FileExistsError:
        pass
    return
  
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
