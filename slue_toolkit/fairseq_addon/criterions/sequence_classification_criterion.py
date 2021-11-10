# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
from fairseq.utils import is_xla_tensor
from sklearn.metrics import f1_score
from functools import partial


@dataclass
class SequenceClassificationCriterionConfig(FairseqDataclass):
    report_f1: bool = field(default=True, metadata={"help": "report f1 scores"})
    class_weights: Optional[List[float]] = field(
        default=None, metadata={"help": "class weights"}
    )


# add slue_ as prefix of the registerred name in case there are conflicts in future
@register_criterion(
    "slue_sequence_classification", dataclass=SequenceClassificationCriterionConfig
)
class SequenceClassificationCriterion(FairseqCriterion):
    def __init__(self, cfg: SequenceClassificationCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.cfg = cfg
        self.num_classes = len(task.label2id)
        self.class_weights = (
            None if cfg.class_weights is None else torch.tensor(cfg.class_weights)
        )
        print("class weights:", self.class_weights)

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(net_output["pooled"])
        loss = F.cross_entropy(
            net_output["pooled"],
            sample["target"],
            reduce=False,
            weight=self.class_weights,
        )
        pred = net_output["pooled"].argmax(dim=-1)

        correct = (pred == sample["target"]).long().sum()
        sample_size = loss.numel()
        if reduce:
            loss = loss.sum()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            # "ntokens": sample_size['net_input']['source'].numel(),
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "correct": correct.item(),
        }

        if self.cfg.report_f1:
            # logging_output['pred'] = pred.cpu()
            # logging_output['target'] = sample['target'].cpu()
            for i in range(self.num_classes):
                p = pred == i
                g = sample["target"] == i
                logging_output[f"_class{i}_tp"] = sum(p & g)  # TP
                logging_output[f"_class{i}_fn"] = sum(~p & g)  # FN
                logging_output[f"_class{i}_fp"] = sum(p & ~g)  # FP
                logging_output[f"_class{i}_tn"] = sum(~p & ~g)  # TN
            logging_output["_num_classes"] = self.num_classes

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        # ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        # metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar(
            "accuracy", correct * 100.0 / sample_size, sample_size, round=3
        )

        if "_num_classes" in logging_outputs[0]:
            num_classes = logging_outputs[0]["_num_classes"]
            for i in range(num_classes):
                for count_type in ["tp", "fp", "fn", "tn"]:
                    name = f"_class{i}_{count_type}"
                    metrics.log_scalar(
                        name, sum(log.get(name, 0) for log in logging_outputs)
                    )

            def compute_class_f1(meters, i):
                if meters[f"_class{i}_tp"].sum == 0:
                    return 0.0
                r = meters[f"_class{i}_tp"].sum / (
                    meters[f"_class{i}_tp"].sum + meters[f"_class{i}_fn"].sum
                )
                p = meters[f"_class{i}_tp"].sum / (
                    meters[f"_class{i}_tp"].sum + meters[f"_class{i}_fp"].sum
                )
                f1 = 2 * r * p / (r + p)
                return f1

            def compute_weighted_f1(meters):
                class_f1s = torch.zeros(num_classes)
                weights = torch.zeros(num_classes)
                for i in range(num_classes):
                    class_f1s[i] = compute_class_f1(meters, i)
                    weights[i] = (
                        meters[f"_class{i}_tp"].sum + meters[f"_class{i}_fn"].sum
                    )
                return (class_f1s * weights).sum().item() / weights.sum().item()

            def compute_macro_f1(meters):
                class_f1s = torch.zeros(num_classes)
                for i in range(num_classes):
                    class_f1s[i] = compute_class_f1(meters, i)
                return class_f1s.mean().item()

            metrics.log_derived("macro_f1", compute_macro_f1)
            metrics.log_derived("weighted_f1", compute_weighted_f1)

            for i in range(num_classes):
                metrics.log_derived(f"f1-cls{i}", partial(compute_class_f1, i=i))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
