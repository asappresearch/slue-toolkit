# Copyright (c) ASAPP Inc.
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import torch

from dataclasses import dataclass

from fairseq.data import encoders
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from fairseq.tasks import register_task
from slue_toolkit.fairseq_addon.data.add_label_dataset import AddLabelDataset


logger = logging.getLogger(__name__)


@dataclass
class AudioClassificationConfig(AudioPretrainingConfig):
    pass


# add slue_ as prefix of the registerred name in case there are conflicts in future
@register_task("slue_audio_classification", dataclass=AudioClassificationConfig)
class AudioClassificationTask(AudioPretrainingTask):
    """ """

    cfg: AudioClassificationConfig

    def __init__(
        self, cfg: AudioClassificationConfig,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"

        self.state.add_factory("label2id", self.load_label2id)

    def load_label2id(self):
        assert self.cfg.labels
        dict_path = os.path.join(self.cfg.data, f"labels.{self.cfg.labels}.txt")
        with open(dict_path) as f:
            labels = [line.strip() for line in f]
        label2id = {l: i for i, l in enumerate(labels)}
        return label2id

    def load_dataset(
        self, split: str, task_cfg: AudioClassificationConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        data_path = self.cfg.data
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        logger.info(f"label2id: {self.label2id}")
        with open(label_path, "r") as f:
            labels = [
                self.label2id[l.strip()]
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        self.datasets[split] = AddLabelDataset(self.datasets[split], labels,)

    @property
    def label2id(self):
        return self.state.label2id

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
