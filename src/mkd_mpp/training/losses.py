from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_loss(prediction: torch.Tensor, target: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "classification":
        return F.binary_cross_entropy_with_logits(prediction, target)
    if task_type == "regression":
        return F.l1_loss(prediction, target)
    raise ValueError(f"Unsupported task type: {task_type}")


@dataclass
class MKDLoss:
    task_type: str
    lambda_kd: float = 1.0
    lambda_cross: float = 0.1
    lambda_cf: float = 0.0
    temperature: float = 1.0

    def __call__(
        self,
        student_output: dict,
        teacher_outputs: dict[str, dict] | None,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        task = supervised_loss(student_output["prediction"], targets, self.task_type)
        kd = self._representation_kd(student_output, teacher_outputs)
        cross = self._cross_modal_alignment(student_output)
        cf = torch.zeros_like(task)
        total = task + self.lambda_kd * kd + self.lambda_cross * cross + self.lambda_cf * cf
        return {"total": total, "task": task, "kd": kd, "cross": cross, "cf": cf}

    def _representation_kd(self, student_output: dict, teacher_outputs: dict[str, dict] | None):
        if not teacher_outputs:
            return torch.zeros_like(student_output["prediction"]).mean()
        losses = []
        for name, teacher_output in teacher_outputs.items():
            if name not in student_output["modal_representations"]:
                continue
            student_rep = student_output["modal_representations"][name]
            teacher_rep = teacher_output["representation"].detach()
            losses.append(F.mse_loss(student_rep, teacher_rep))
        if not losses:
            return torch.zeros_like(student_output["prediction"]).mean()
        return torch.stack(losses).mean()

    def _cross_modal_alignment(self, student_output: dict):
        reps = list(student_output["modal_representations"].values())
        if len(reps) < 2:
            return torch.zeros_like(student_output["prediction"]).mean()
        losses = []
        for i, left in enumerate(reps):
            for right in reps[i + 1 :]:
                losses.append(F.mse_loss(nn.functional.normalize(left, dim=-1), nn.functional.normalize(right, dim=-1)))
        return torch.stack(losses).mean()
