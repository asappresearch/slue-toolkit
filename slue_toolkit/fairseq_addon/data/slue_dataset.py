import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
import librosa

logger = logging.getLogger(__name__)


class SlueAudioClassificationDataset(RawAudioDataset):
    """Extend FileAudioFeatDataset to have classification labels"""

    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
        )
        self.fnames = []
        self.labels = []

        labels = open(f"{os.path.dirname(manifest_path)}/labels.txt").readlines()
        label2id = {l.strip(): i for i, l in enumerate(labels)}

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            manifest_prefix = manifest_path.split(".tsv")[0]
            labels_list = open(f"{manifest_prefix}.sent").readlines()
            for line, label in zip(f, labels_list):
                label = label.strip()
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = float(items[1])
                if max_sample_size is not None:
                    sz = min(sz, max_sample_size)
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    continue

                self.fnames.append(items[0])
                self.sizes.append(sz)
                self.labels.append(label2id[label])
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = librosa.load(fname, sr=self.sample_rate)
        assert curr_sample_rate == self.sample_rate, curr_sample_rate
        feats = torch.from_numpy(wav).float()
        # crop or pad

        out = {"id": index, "target": self.labels[index]}
        feats = self.postprocess(feats, curr_sample_rate)
        out["source"] = feats
        return out

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask

        print(samples)
        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "target": torch.LongTensor([s["target"] for s in samples]),
            "net_input": input,
        }
