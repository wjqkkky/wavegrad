pip install .
python -m wavegrad.preprocess ../../fred/waveglow/LibriTTS/train-clean-100/
CUDA_VISIBLE_DEVICES=6 python -m wavegrad ../wavegrad-checkpoints/ ../../fred/waveglow/LibriTTS/train-clean-100/

