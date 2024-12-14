# EnhancedSGD: Adaptive Optimizer with Q-Learning

**EnhancedSGD** is a PyTorch-based optimizer that extends the classic Stochastic Gradient Descent (SGD) algorithm with **adaptive learning rate**, **momentum adjustments**, and **gradient processing enhancements**. This optimizer incorporates **Q-Learning** to dynamically adjust parameters based on training feedback, making it highly suitable for complex deep learning tasks such as large language models (LLMs).

---

## Features

- **Q-Learning Integration**: Utilizes a Q-Learning controller for dynamic parameter adjustments, including learning rate and momentum scaling.
- **Entropy-Based Adjustments**: Adjusts gradients and learning rate based on entropy and variance metrics for more stable optimization.
- **Gradient Retention**: Implements gradient buffering for entropy-driven retention of critical gradients.
- **Lookahead Optimization**: Improves convergence by looking ahead at parameter updates with tunable steps and blending factors.
- **Adaptive Clipping**: Dynamically clips gradients based on variance to prevent exploding gradients.
- **Loss Spike Detection**: Automatically reduces learning rate and momentum in response to significant loss spikes.
- **Noise Application**: Adds controlled noise to gradients for regularization during training.
- **Bayesian Parameter Initialization**: Provides an option for initializing parameters using Bayesian techniques for robustness.
- **Flexible Configuration**: Supports fine-tuning for various deep learning use cases, including LLM training.

---

## Installation

Clone the repository and ensure you have the required dependencies installed:

```bash
git clone https://github.com/waefrebeorn/EnhancedSGD.git
cd EnhancedSGD
```

Dependencies:
- Python 3.8+
- PyTorch 1.10+ with CUDA support
- NumPy
- SciPy

---

## Usage

### Import and Initialize

```python
from EnhancedSGD import EnhancedSGD
import torch.nn as nn

# Example model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Define optimizer
optimizer = EnhancedSGD(
    params=model.parameters(),
    lr=0.01,
    momentum=0.9,
    smoothing_factor=0.1,
    entropy_weight=0.1,
    adaptive_momentum=True
)

# Training loop
for epoch in range(1, 101):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), target)
        loss.backward()
        optimizer.step(closure=lambda: loss, current_epoch=epoch)
```

### Q-Learning Customization

You can customize the Q-Learning controller's behavior via:
- Learning rate (`learning_rate`)
- Discount factor (`discount`)
- Exploration strategy (`epsilon`, `epsilon_decay`)
- Action space mix probability (`initial_mix_prob`)

Example:
```python
optimizer.q_controller.update_mix_prob(current_loss, epoch)
```

---

## Key Hyperparameters

- **`lr`**: Initial learning rate.
- **`momentum`**: Initial momentum value.
- **`adaptive_momentum`**: Whether to enable momentum adaptation via Q-Learning.
- **`entropy_weight`**: Weight of entropy in Q-Learning reward calculations.
- **`gradient_clipping`**: Enable adaptive gradient clipping.
- **`lookahead_k`**: Number of lookahead steps for weight updates.
- **`lookahead_alpha`**: Step size for lookahead blending.
- **`loss_correction_factor`**: Factor to reduce learning rate upon loss spikes.
- **`gradient_centering`**: Center gradients by subtracting their mean.

---

## Advanced Features

### Entropy-Based Adjustments

EnhancedSGD calculates the entropy of the gradient distribution to stabilize updates and adjust the optimizer's parameters dynamically.

### Gradient Retention

Critical gradients are retained in a buffer for reuse during training. The size of this buffer is configurable via `gradient_buffer_size`.

### Lookahead Optimization

Lookahead steps blend fast and slow weights for smoother convergence.

---

## Example Projects

EnhancedSGD has been tested on tasks like:
1. Large Language Models (LLMs)
2. Image classification (e.g., MNIST, CIFAR)
3. Reinforcement Learning with policy optimization

---

## References

This work builds on:
- **Stochastic Gradient Descent (SGD)**: https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
- **Q-Learning**: Watkins, C. J. C. H., & Dayan, P. (1992). Q-Learning. *Machine Learning, 8*, 279–292. [DOI:10.1007/BF00992698](https://doi.org/10.1007/BF00992698)
- **Lookahead Optimizer**: Zhang, M., & Mitliagkas, I. (2019). Lookahead Optimizer: k steps forward, 1 step back. [arXiv:1907.08610](https://arxiv.org/abs/1907.08610)

---

## Contributing

We welcome contributions! Please open an issue or submit a pull request with your improvements.

---

## License

MIT License. See `LICENSE` for details.
