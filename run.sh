pip install .
python -m wavegrad.preprocess ../datasets/test-wavegrad/
CUDA_VISIBLE_DEVICES=6 python -m wavegrad  ../wavegrad-checkpoints-test ../datasets/test-wavegrad/