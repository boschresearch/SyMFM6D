from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import math


def save_div(a, b):
    return a/b if b != 0 else 1


class ValidationIntervalScheduler(Callback):
    def __init__(self, tot_iter, clr_div):
        super().__init__()
        self.tot_iter = tot_iter
        self.clr_div = clr_div

    def is_to_eval(self, it):
        # tot_iter := num_epochs * num_iter_per_epoch
        # ckr_div := cyclic learning rate divisor (default: 2)
        if it == 100:
            return True, 1
        wid = self.tot_iter // self.clr_div
        if (it // wid) % 2 == 1:
            eval_frequency = wid // 15
        else:
            eval_frequency = wid // 9
        return eval_frequency

    def on_epoch_start(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        val_interval = min(self.is_to_eval(trainer.global_step), trainer.num_training_batches)
        f = math.ceil(save_div(trainer.num_training_batches, val_interval))
        print(f"Model will be evaluated {f} times this epoch")
        trainer.val_check_batch = int(trainer.num_training_batches * (1/f))
