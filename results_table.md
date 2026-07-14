| dataset (seeds) | corruption | broad-aug ff | +TTA | +MEMO | +imagination | verdict |
|---|---|---|---|---|---|---|
| MNIST (5) | band | 0.599±0.027 | 0.410±0.017 | 0.599±0.027 | **0.659±0.020** | **imag wins** |
| MNIST (5) | patch | 0.631±0.025 | 0.514±0.015 | 0.630±0.025 | **0.632±0.019** | tie/loss |
| MNIST (5) | scattered | 0.700±0.064 | 0.588±0.063 | 0.700±0.063 | **0.676±0.064** | tie/loss |
| MNIST (5) | noise | 0.416±0.038 | 0.309±0.058 | 0.416±0.038 | **0.349±0.021** | **no win** (neg. control) [ok] |
| Fashion (5) | band | 0.607±0.024 | 0.508±0.019 | 0.607±0.024 | **0.643±0.013** | **imag wins** |
| Fashion (5) | patch | 0.562±0.012 | 0.461±0.006 | 0.563±0.012 | **0.630±0.011** | **imag wins** |
| Fashion (5) | scattered | 0.675±0.018 | 0.583±0.019 | 0.675±0.017 | **0.623±0.014** | tie/loss |
| Fashion (5) | noise | 0.279±0.036 | 0.218±0.033 | 0.279±0.035 | **0.225±0.020** | **no win** (neg. control) [ok] |
| CIFAR-10 (5) | band | 0.383±0.032 | 0.365±0.028 | 0.382±0.034 | **0.399±0.019** | tie/loss |
| CIFAR-10 (5) | patch | 0.363±0.021 | 0.347±0.024 | 0.362±0.021 | **0.409±0.025** | **imag wins** |
| CIFAR-10 (5) | scattered | 0.245±0.010 | 0.229±0.010 | 0.242±0.011 | **0.567±0.012** | **imag wins** |
| CIFAR-10 (5) | noise | 0.452±0.007 | 0.462±0.022 | 0.450±0.009 | **0.462±0.011** | **no win** (neg. control) [ok] |

### one-line rule outcome
- imagination wins (5 cells): MNIST/band (+0.059), Fashion/band (+0.036), Fashion/patch (+0.068), CIFAR-10/patch (+0.046), CIFAR-10/scattered (+0.322)
- noise negative control holds everywhere: True
