import logging

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .wav2vec2.model import wav2vec2_model

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        ckpt = torch.load(ckpt)
        self.model = wav2vec2_model(**ckpt["config"])
        result = self.model.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info(f"missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")
        logger.info(f"{sum(p.numel() for p in self.model.parameters())} params")

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)

        # NOTE: the model knows whether to normalize waves before conv feature extractor
        hs = self.model.extract_features(
            pad_sequence(wavs, batch_first=True),
            wav_lengths,
        )[0]

        return {
            "hidden_states": hs,
        }