# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import BaseWrapperDataset, data_utils


class AddLabelDataset(BaseWrapperDataset):
    def __init__(
        self, dataset, labels,
    ):
        super().__init__(dataset)
        self.labels = labels
        assert len(self.labels) == len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.labels[index]
        return item

    def size(self, index):
        return self.dataset.size(index)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]
        if type(target[0]) == int:
            target = torch.LongTensor(target)
        elif type(target[0]) == list:
            target = torch.FloatTensor(target)
        collated["target"] = target

        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
