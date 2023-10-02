import numpy as np
import torch


# アーリーストッピングを行うクラス
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    based on: https://github.com/Bjarten/early-stopping-pytorch
    """

    # init関数
    def __init__(self, patience=3, verbose=False, delta=0, text_path=None):
        # 早期終了をかける際に許容するepoch数
        self.patience = patience
        # 損失値の改善をprintで表示するか否かの指定
        self.verbose = verbose
        # 損失値が改善しないまま経過したepochを記録
        self.counter = 0
        # ここに最善の損失値の結果が蓄積される
        self.best_score = None
        self.val_loss_min = np.Inf
        # これがTrueになると早期終了がかかる
        self.early_stop = False
        # 基本使わない変数
        self.delta = delta
        # ログを書き込む際のテキストフォルダのパス
        self.text_path = text_path

    # call関数
    def __call__(
        self,
        val_loss=None,
        model=None,
        rem_epochs=None,
        optimizer=None,
        log_train_train=None,
        log_train_eval=None,
        log_valid_eval=None,
        save_path=None,
    ):

        # 入力された損失値
        score = -val_loss

        """
        損失値の改善の判断を行う処理---------------------------
        """
        # 過去の最善の値が無い時の処理
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                val_loss=val_loss,
                model=model,
                rem_epochs=rem_epochs,
                optimizer=optimizer,
                log_train_train=log_train_train,
                log_train_eval=log_train_eval,
                log_valid_eval=log_valid_eval,
                best_score=self.best_score,
                save_path=save_path,
            )

        # 　学習時に損失値の改善が見られなかった場合の処理
        elif score < self.best_score + self.delta:
            # カウンターを1つ増やす
            self.counter += 1
            # 経過のprint表示
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            with open(self.text_path, "a", encoding="utf_8_sig") as f:
                f.write(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}\n"
                )
            # もし、カウンターがpatienceと同じになれば、学習の早期終了を行う
            if self.counter >= self.patience:
                self.early_stop = True

        # 学習時の損失値が改善した場合の処理
        else:
            self.best_score = score
            # チェックポイントの保存
            self.save_checkpoint(
                val_loss=val_loss,
                model=model,
                rem_epochs=rem_epochs,
                optimizer=optimizer,
                log_train_train=log_train_train,
                log_train_eval=log_train_eval,
                log_valid_eval=log_valid_eval,
                best_score=self.best_score,
                save_path=save_path,
            )
            # 　カウンターを初期化
            self.counter = 0

    # チェックポイントの保存に使う関数
    def save_checkpoint(
        self,
        val_loss=None,
        model=None,
        rem_epochs=None,
        epoch_iter_step=None,
        optimizer=None,
        log_train_train=None,
        log_train_eval=None,
        log_valid_eval=None,
        best_score=None,
        save_path=None,
    ):
        # 推移のprint表示
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} -->\
                    {val_loss:.6f}).  Saving model ..."
            )
            with open(self.text_path, "a", encoding="utf_8_sig") as f:
                f.write(
                    f"Validation loss decreased ({self.val_loss_min:.6f}\
                        --> {val_loss:.6f}).  Saving model ...\n"
                )
        # チェックポイントの保存
        torch.save(
            {
                "epoch": rem_epochs,
                "step": epoch_iter_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "log_train_train": log_train_train,
                "log_train_eval": log_train_eval,
                "log_valid_eval": log_valid_eval,
                "best_score": best_score,
                "early_stop": self.early_stop,
            },
            save_path,
        )
        self.val_loss_min = val_loss
