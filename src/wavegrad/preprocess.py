# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from wavegrad.params import params

from wavegrad.mozilla_tts_audio import AudioProcessor

ap = AudioProcessor(                 
     sample_rate=params.sample_rate,
     num_mels=params.num_mels,
     min_level_db=params.min_level_db,
     frame_shift_ms=params.frame_length_ms,
     frame_length_ms=params.frame_length_ms,
     hop_length=params.hop_length,
     win_length=params.win_length,
     ref_level_db=params.ref_level_db,
     fft_size=params.fft_size,
     power=params.power,
     preemphasis=params.preemphasis,
     signal_norm=params.signal_norm,
     symmetric_norm=params.symmetric_norm,
     max_norm=params.max_norm,
     mel_fmin=params.mel_fmin,
     mel_fmax=params.mel_fmax,
     spec_gain=params.spec_gain,
     stft_pad_mode=params.stft_pad_mode,
     clip_norm=params.clip_norm,
     griffin_lim_iters=params.griffin_lim_iters,
     do_trim_silence=params.do_trim_silence,
     trim_db=params.trim_db)

def transform(filename):
  audio, sr = T.load_wav(filename)
  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')
  audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)

  #hop = params.hop_length
  #win = hop * 4
  #n_fft = 2**((win-1).bit_length())
  #f_max = sr / 2.0
  #mel_spec_transform = TT.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=True)

  with torch.no_grad():
    #spectrogram = mel_spec_transform(audio)
    #spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    #spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    spectrogram = np.float32(ap.melspectrogram(audio.detach().cpu().numpy()))
    np.save(f'{filename}.spec.npy', spectrogram)


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train WaveGrad')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())
