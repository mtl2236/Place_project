# DeepPlace

An end-to-end learning approach DeepPlace for placement problem. The deep reinforcement learning (DRL) agent places the macros sequentially. We use [PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) for all the experiments implemented with Pytorch.

## Requirements

* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)


In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines-master
pip install -e .

# Other requirements
pip install -r requirements.txt


# DGL installation
conda install -c dglteam dgl-cuda10.2
```

## Training
python  main.py   InputDataSample/sample50_compact/50-99/placement_info.txt    InputDataSample/connect_file/connect_50.txt


## Results
vi reult_info.txt

