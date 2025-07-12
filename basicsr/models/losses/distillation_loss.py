import torch
import torch.nn as nn
# from basicsr.utils.registry import LOSS_REGISTRY

# @LOSS_REGISTRY.register()
class DistillationLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', loss_student='L1Loss', loss_distill='L1Loss', alpha=0.5, beta=0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta

        self.student_loss_fn = getattr(nn, loss_student)()
        self.distill_loss_fn = getattr(nn, loss_distill)()
        self.reduction = reduction

    def forward(self, student_pred, teacher_pred, gt):
        loss_student = self.student_loss_fn(student_pred, gt)
        loss_distill = self.distill_loss_fn(student_pred, teacher_pred)
        return self.loss_weight * (self.alpha * loss_student + self.beta * loss_distill)
