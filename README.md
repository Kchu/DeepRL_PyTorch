# Deep Reinforcement Learning Codes
Currently there are only the codes for distributional reinforcement learning here. Codes for algorithms: DQN, C51, QR-DQN, IQN, QUOTA.

The codes for C51, QR-DQN, and IQN are a slight change from [sungyubkim](<https://github.com/sungyubkim/Deep_RL_with_pytorch/tree/master/6_Uncertainty_in_RL>). QUOTA is implemented based on the work from the algorithm's author: [Shangtong Zhang](<https://github.com/ShangtongZhang>). I recently noticed that my DQN code may not get an ideal performance, while other codes run well. I would much appreciate it if someone could point out the errors in my code.

Always up for a chat -- shoot me an email if you'd like to discuss anything.

## Dependency:

* pytorch(>=1.0.0)
* gym(=0.10.9)
* numpy
* matplotlib

## Usage:

When your computer's python environment satisfies the above dependencies, you can run the code. For example, enter:
```python
python 3_iqn.py Breakout 
```
on the command line to run the algorithms in the Atari environment.
You can change some specific parameters for the algorithms inside the codes.

After training, you can plot the results by running result_show.py with appropriate parameters.

## References:

1. Human-level control through deep reinforcement learning(DQN)   [[Paper](https://www.nature.com/articles/nature14236)]   [[Code](https://github.com/Kchu/DeepRL_CK/blob/master/Distributional_RL/0_DQN.py)]

2. A Distributional Perspective on Reinforcement Learning(C51)   [[Paper](https://arxiv.org/abs/1707.06887v1)]   [[Code](https://github.com/Kchu/DeepRL_CK/blob/master/Distributional_RL/1_C51.py)]

3. Distributional Reinforcement Learning with Quantile Regression(QR-DQN)   [[Paper](https://arxiv.org/abs/1710.10044v1)]   [[Code](https://github.com/Kchu/DeepRL_CK/blob/master/Distributional_RL/2_QR_DQN.py)]

4. Implicit Quantile Networks for Distributional Reinforcement Learning(IQN)   [[Paper](https://arxiv.org/abs/1806.06923v1)]   [[Code](https://github.com/Kchu/DeepRL_CK/blob/master/Distributional_RL/3_IQN.py)]

5. QUOTA: The Quantile Option Architecture for Reinforcement Learning(QUOTA)  [[Paper](https://arxiv.org/abs/1811.02073v2)]   [[Code](https://github.com/Kchu/DeepRL_CK/blob/master/Distributional_RL/4_QUOTA.py)]
