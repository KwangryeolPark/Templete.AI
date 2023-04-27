## lookahead optimizer for pytorch
![License](https://img.shields.io/github/license/alphadl/lookahead.pytorch)
![Star](https://img.shields.io/github/stars/alphadl/lookahead.pytorch)
![Fork](https://img.shields.io/github/forks/alphadl/lookahead.pytorch)
![Twitter](https://img.shields.io/twitter/url/https/github.com%2Falphadl%2Flookahead.pytorch)

PyTorch implement of <a href="https://arxiv.org/abs/1907.08610" target="_blank">Lookahead Optimizer: k steps forward, 1 step back</a>

Usage:
```python
base_opt = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999)) # Any optimizer
lookahead = Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead
lookahead.zero_grad()
loss_function(model(input), target).backward() # Self-defined loss function
lookahead.step()
```
