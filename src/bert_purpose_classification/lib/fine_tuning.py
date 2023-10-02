import time

import torch
import torch.optim as optim
from lib.EarlyStopping import EarlyStopping


# モデルのファインチューニング用の関数
def fine_tuning(
    model=None,
    dataloaders_dict=None,
    num_epochs=10000,
    learning_rate=2e-5,
    checkpoint_path=None,
    restart=None,
    save_epoch_for_interval_model=None,
    use_vector="cls",
    loss_weight=None,
    study_early_stop=None,
    patience=None,
    print_batch_log=None,
    device=None,
    batch_size=2,
):
    """
    Args:
        model: 対象のBERTモデル
        dataloaders_dict: notebook上部で定義したdataloaders_dict(型はlong)
        optimizer: オプティマイザ
        num_epochs: 学習させたいエポック数(デフォルトは10000)
        checkpoint_path：チェックポイントを保存するpath
            （チェックポイントを読み込む時も、このpathを使用）
        restart: チェックポイントから学習を再開するかどうかを指定（Trueで再開）
        save_epoch_for_interval_model:指定した数値の倍数におけるepochで，別途チェックポイントを保存。
        use_vector: 分類層に入力するベクトルの種類を指定（デフォルトは'cls'）
                    'cls': BERTモデルから出力されるCLSトークンに対応したベクトルのみをを用いて推論
                    'avg_pooling': モデルから出力される全てのトークンのベクトルの平均を取った768次元ベクトルを用いて推論
        early_stop：学習時に早期終了をするか否かを指定（デフォルトはNone）
        patience: 早期終了の際に，許容するエポック数
    """

    """
    前準備（共通）---------------------------------------------------------
    """
    # 学習における全てのepoch数を定義
    total_epochs = num_epochs
    # Optimizerの設定（学習率はここで決める）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 学習の経過を保存するテキストファイルのpath
    text_path = checkpoint_path.replace(".pt", "_log.txt")
    # 早期終了の設定
    if study_early_stop is not None:
        early_stopping = EarlyStopping(
            patience=patience, verbose=True, text_path=text_path
        )

    """
    -----------------------------------------------------------------------------------
    """
    if restart is None:
        """
        前準備（最初に訓練を始める時）-----------------------------------------
        """
        # モデルをgpuへ
        model.to(device)
        torch.backends.cudnn.benchmark = True
        # 訓練時のログを格納する空のリストを定義
        log_train_train = []
        log_train_eval = []
        log_valid_eval = []
        # ログを書き込むテキストファイル作成
        with open(text_path, "w", encoding="utf_8_sig", newline="") as f:
            f.write("Epoch {}/{} | {:^5} | Loss:{:.4f} Acc:{:.4f} time:{:.4f}")
        # モデルを検証モードに
        model.eval()

    else:
        """
        前準備（チェックポイントから訓練を再開する場合）----------------------
        """
        # モデルをGPUへ
        model.to(device)
        torch.backends.cudnn.benchmark = True
        # チェックポイントの読み込み
        checkpoint = torch.load(checkpoint_path)
        # チェックポイントにおけるモデルのパラメータの読み込み
        model.load_state_dict(checkpoint["model_state_dict"])
        # チェックポイントにおけるオプティマイザの状態の読み込み
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # 残りのepoch数の読み込み
        num_epochs = checkpoint["epoch"]
        # それまでのログの読み込み
        log_train_train = checkpoint["log_train_train"]
        log_train_eval = checkpoint["log_train_eval"]
        log_valid_eval = checkpoint["log_valid_eval"]
        # 早期終了を行う場合
        if study_early_stop is not None:
            best_score = checkpoint["best_score"]
            early_stopping.best_score = best_score
        # 学習においてもっとも良い（低い）損失値の読み込み
        # モデルを検証モードに
        model.eval()
    """
    ファインチューニング開始-----------------------------------------------
    """
    # 指定されたepochの数だけ繰り返すfor文
    for epoch in range(1, num_epochs + 1):
        # 開始時刻の記録
        s_time = time.time()

        # 各フェイズ毎を処理を行うためのfor文
        """
        フェイズの説明：
        train_train: 訓練データを使ってファインチューニングを行う（誤差逆伝播あり）
        train_eval: 訓練データを使って検証を行う（誤差逆伝播なし）
        valid_eval: 検証データを使って検証を行う（誤差逆伝播なし）
        """
        for phase in ["train_train", "train_eval", "valid_eval"]:
            if phase == "train_train":
                # モデルを訓練モードに
                model.train()
            else:
                # モデルを検証モードに
                model.eval()
            """
            各種結果を記録するための変数を定義-----------------------------------
            """
            # 損失値の記録用
            epoch_loss = 0.0
            # 正解したデータの数の記録用
            epoch_corrects = 0
            # 経過したミニバッチの数を記録
            batch_processed_num = 0
            """
            ミニバッチからデータを取り出して、学習or推論を行う-------------------
            """
            for batch in dataloaders_dict[phase]:
                """
                各入力idを定義----------------------------------------------------
                """
                # トークンid
                input_token_ids = batch[0].to(device)
                # マスクid
                input_mask_ids = batch[1].to(device)
                # セグメントid
                input_segment_ids = batch[2].to(device)
                # 分類ラベル
                input_labels = batch[3].to(device)
                """
                各id列をモデルに入力し、推論、損失値計算、誤差逆伝播----------------
                """
                # optimizerの初期化
                optimizer.zero_grad()
                # phaseが'train_train'の時はモデルパラメータの更新をTrueに
                # (おまじないコード？)
                with torch.set_grad_enabled(phase == "train_train"):
                    # BERTモデルへ各id列を入力
                    results = model(
                        input_ids=input_token_ids,
                        attention_mask=input_mask_ids,
                        token_type_ids=input_segment_ids,
                        labels=input_labels,
                        output_vec=use_vector,
                        loss_weight=loss_weight,
                    )
                    # 予測ラベルの取得
                    _, preds = torch.max(results["logits"], 1)
                    preds = preds.view(-1)
                    # 損失値の取得
                    loss = results["loss"]
                    # phaseが「train_train」の場合
                    if phase == "train_train":
                        # バックプロパゲーション（誤差逆伝播）
                        loss.backward()
                        optimizer.step()
                    # 損失値の記録
                    curr_loss = loss.item()
                    epoch_loss += curr_loss
                    # それぞれの予測ラベルにおいて正解した数を蓄積
                    epoch_corrects += torch.sum(preds == input_labels.data)
                # ミニバッチをすすめる
                batch_processed_num += 1
                # 10バッチ経過する毎にログを表示する
                if print_batch_log:
                    if batch_processed_num % 10 == 0 and batch_processed_num != 0:
                        print(
                            "Processed : ",
                            batch_processed_num * batch_size,
                            " Loss : ",
                            curr_loss,
                        )
            """
            epoch全体における学習・推論結果をまとめる-----------------------------
            """
            # エポックにおける平均損失値と正解率を計算
            epoch_loss = epoch_loss / (len(dataloaders_dict[phase].dataset))
            epoch_acc = epoch_corrects.double() / (len(dataloaders_dict[phase].dataset))
            # ログへの格納
            if phase == "train_train":
                log_train_train.append([epoch_loss, epoch_acc])
            elif phase == "train_eval":
                log_train_eval.append([epoch_loss, epoch_acc])
            else:
                log_valid_eval.append([epoch_loss, epoch_acc])
            # 終了時刻の記録
            e_time = time.time()
            study_time = e_time - s_time
            # 残りのエポック数を記録
            rem_epochs = num_epochs - epoch
            # 現在のepoch数を記録
            now_epoch = total_epochs - rem_epochs
            # ログの表示
            print(
                "Epoch {}/{} | {:^5} | Loss:{:.4f} Acc:{:.4f} time:{:.4f}\n".format(
                    now_epoch, total_epochs, phase, epoch_loss, epoch_acc, study_time
                )
            )
            # ログをテキストファイルに書き込む
            with open(text_path, "a", encoding="utf_8_sig") as f:
                f.write(
                    "Epoch {}/{} | {:^5} | Loss:{:.4f} Acc:{:.4f} time:{:.4f}\n".format(
                        now_epoch,
                        total_epochs,
                        phase,
                        epoch_loss,
                        epoch_acc,
                        study_time,
                    )
                )

            """
            検証データに対する推論が終わった場合のみ実行される処理-----------------------
            """
            if phase == "valid_eval":
                # 早期終了をかけるかどうかの判断
                early_stopping(
                    val_loss=epoch_loss,
                    model=model,
                    rem_epochs=rem_epochs,
                    optimizer=optimizer,
                    log_train_train=log_train_train,
                    log_train_eval=log_train_eval,
                    log_valid_eval=log_valid_eval,
                    save_path=checkpoint_path,
                )
                # 早期終了がかかった場合
                if early_stopping.early_stop:
                    print("Early stopping")
                    # 学習全体におけるログを保存
                    torch.save(
                        {
                            "log_train_train": log_train_train,
                            "log_train_eval": log_train_eval,
                            "log_valid_eval": log_valid_eval,
                            "patience": patience,
                            "early_stop": early_stopping.early_stop,
                        },
                        checkpoint_path.replace(".pt", ".all_logs"),
                    )
                    # キャッシュを削除
                    torch.cuda.empty_cache()
                    return

                # 指定した倍数のepochでモデルを別途保存するための関数
                if (
                    save_epoch_for_interval_model is not None
                    and now_epoch % save_epoch_for_interval_model == 0
                ):
                    torch.save(
                        {
                            "epoch": rem_epochs,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        checkpoint_path.replace(".pt", f"_{now_epoch}" + "epoch.pt"),
                    )

    """
    早期終了がかからずに学習が終了した場合の処理---------------------------------------
    """
    # その時のチェックポイントを保存
    torch.save(
        {
            "epoch": rem_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "log_train_train": log_train_train,
            "log_train_eval": log_train_eval,
            "log_valid_eval": log_valid_eval,
            "early_stop": early_stopping.early_stop,
        },
        checkpoint_path,
    )
    # 学習全体のログを保存
    torch.save(
        {
            "log_train_train": log_train_train,
            "log_train_eval": log_train_eval,
            "log_valid_eval": log_valid_eval,
            "patience": patience,
            "early_stop": early_stopping.early_stop,
        },
        checkpoint_path.replace(".pt", ".all_logs"),
    )
    # キャッシュを削除
    torch.cuda.empty_cache()
    return
