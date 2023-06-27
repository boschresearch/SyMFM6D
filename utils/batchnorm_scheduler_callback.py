from pytorch_lightning.callbacks import Callback
import torch.nn as nn


class BNMomentumScheduler(Callback):
    def __init__(self, model, bn_lambda, last_epoch=-1, accumulate_grad_batches=1):
 

        self.model = model
        self.setter = self.set_bn_momentum_default
        self.lmbd = bn_lambda
        self.accumulate_grad_batches = accumulate_grad_batches

        # self.step(last_epoch + 1)
        # self.last_epoch = last_epoch
    def set_bn_momentum_default(self, bn_momentum):
        def fn(m):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = bn_momentum

        return fn

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        momentum = max(
            0.9 * 0.5 ** (int(trainer.global_step * self.accumulate_grad_batches * max(batch['rgb'].shape[0], 5) / 2e5)),
            1e-2,
        )
        self.model.apply(self.setter(self.lmbd(trainer.current_epoch)))
        trainer.logger.log_metrics({"batch_norm_momentum": momentum}, step=trainer.global_step)

    # def step(self, epoch=None):
    #     if epoch is None:
    #         epoch = self.last_epoch + 1

    #     self.last_epoch = epoch
    #     self.model.apply(self.setter(self.lmbd(epoch)))
    #     trainer.logger.log_metrics({"batch_norm_momentum": momentum}, step=trainer.global_step)


# class BatchNormMomentumScheduler(Callback):
#     def __init__(self, accumulate_grad_batches):
#         self.setter = self.set_bn_momentum_default
#         self.accumulate_grad_batches = accumulate_grad_batches

#     def set_bn_momentum_default(self, bn_momentum):
#         def fn(m):
#             if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
#                 m.momentum = bn_momentum

#         return fn

#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
#         momentum = max(
#             0.9 * 0.5 ** (int(trainer.global_step * self.accumulate_grad_batches * max(batch['rgb'].shape[0], 5) / 2e5)),
#             1e-2,
#         )
#         pl_module.apply(self.setter(momentum))
#         trainer.logger.log_metrics({"batch_norm_momentum": momentum}, step=trainer.global_step)

