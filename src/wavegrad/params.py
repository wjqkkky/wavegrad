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


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    else:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=30,
    learning_rate=2e-4,
    max_grad_norm=1.0,

    # upsample factors
    #factors=[5, 5, 3, 2, 2], # 5*5*3*2*2=300 (hop_lenght=300)
    factors=[4, 4, 4, 2, 2], # 4*4*4*2*2=256 (this is necessary to be equal to hop_lenght)

    # Audio params
    num_mels=80,   
    fft_size=1024,     
    sample_rate=22050, 
    win_length=1024,  
    hop_length=256, # if you change that change factors

    frame_length_ms=None, 
    frame_shift_ms=None,  
    preemphasis=0.98,   
    min_level_db=-100,  
    ref_level_db=20,     
    power=1.5,           
    griffin_lim_iters=60,
    stft_pad_mode="reflect",
    signal_norm=True,    
    symmetric_norm=True, 
    max_norm=4.0,       
    clip_norm=True,  
    mel_fmin=0.0,      
    mel_fmax=8000.0,      
    spec_gain=20.0, 
    do_trim_silence=False,  
    trim_db=60,

    # Data params
    crop_mel_frames=24,
    # Model params
    noise_schedule=np.linspace(1e-6, 0.01, 1000).tolist(),
)
