import io
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _as_mono(audio):
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def _crop_or_pad(waveform, target_samples):
    current = waveform.shape[-1]
    if current == target_samples:
        return waveform
    if current > target_samples:
        start = (current - target_samples) // 2
        return waveform[..., start : start + target_samples]
    return F.pad(waveform, (0, target_samples - current))


class BeansCBIParquetDataset(Dataset):
    """Lazy BEANS-CBI dataset backed by Hugging Face parquet shards.

    The released split CSV files store parquet shard names, row-group indices,
    and labels. Audio bytes are read lazily in ``__getitem__`` so the dataset is
    never decoded fully into memory.
    """

    def __init__(
        self,
        index_csv,
        data_dir,
        label_mapping_csv=None,
        sample_rate=32000,
        duration=10.0,
        mono=True,
        row_group_cache_size=1,
    ):
        self.index_csv = Path(index_csv)
        self.data_dir = Path(data_dir)
        self.frame = pd.read_csv(self.index_csv)
        required = {"parquet_file", "row_group", "row_in_row_group", "label", "label_idx", "sample_id"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Index CSV missing required columns: {sorted(missing)}")

        self.sample_rate = int(sample_rate) if sample_rate is not None else None
        self.duration = duration
        self.mono = bool(mono)
        self.row_group_cache_size = max(1, int(row_group_cache_size))
        self._row_group_cache = OrderedDict()

        if label_mapping_csv:
            mapping = pd.read_csv(label_mapping_csv)
            self.classes = mapping.sort_values("label_idx")["label"].tolist()
        else:
            self.classes = (
                self.frame[["label", "label_idx"]]
                .drop_duplicates()
                .sort_values("label_idx")["label"]
                .tolist()
            )

    def __len__(self):
        return len(self.frame)

    def _resolve_parquet_file(self, value):
        path = Path(str(value))
        if path.is_absolute():
            return path
        return self.data_dir / path

    def _read_row_group(self, parquet_file, row_group):
        parquet_file = self._resolve_parquet_file(parquet_file)
        key = (str(parquet_file), int(row_group))
        if key in self._row_group_cache:
            self._row_group_cache.move_to_end(key)
            return self._row_group_cache[key]

        table = pq.ParquetFile(key[0]).read_row_group(key[1], columns=["path", "label", "split"])
        self._row_group_cache[key] = table
        while len(self._row_group_cache) > self.row_group_cache_size:
            self._row_group_cache.popitem(last=False)
        return table

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        table = self._read_row_group(row["parquet_file"], int(row["row_group"]))
        item = table.slice(int(row["row_in_row_group"]), 1).to_pylist()[0]
        audio_struct = item["path"]
        audio, source_rate = sf.read(io.BytesIO(audio_struct["bytes"]), dtype="float32", always_2d=False)

        if self.mono:
            audio = _as_mono(audio)
        waveform = torch.from_numpy(audio).unsqueeze(0) if audio.ndim == 1 else torch.from_numpy(audio.T)

        if self.sample_rate is not None and int(source_rate) != self.sample_rate:
            try:
                import torchaudio.functional as AF
            except ImportError as exc:
                raise ImportError("torchaudio is required when resampling is requested") from exc
            waveform = AF.resample(waveform, int(source_rate), self.sample_rate)
            source_rate = self.sample_rate

        if self.duration is not None:
            target_samples = int(round(float(self.duration) * int(source_rate)))
            waveform = _crop_or_pad(waveform, target_samples)

        return {
            "waveform": waveform,
            "label_idx": int(row["label_idx"]),
            "label": row["label"],
            "sample_id": row["sample_id"],
            "audio_path": row.get("audio_path", audio_struct.get("path", "")),
        }


def collate_waveforms(batch):
    waveforms = torch.stack([item["waveform"] for item in batch], dim=0)
    labels = torch.as_tensor([item["label_idx"] for item in batch], dtype=torch.long)
    sample_ids = [item["sample_id"] for item in batch]
    return waveforms, labels, sample_ids

