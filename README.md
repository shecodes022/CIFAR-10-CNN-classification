# Implementing a CNN using the CIFAR-10 Dataset 🧠

*Deep Learning and Neural Networks with Applications · Queen Mary University of London*

## Motivation ⚙️

### Chosen data-related problem

> **"How effectively can a custom Convolutional Neural Network classify images across 10 categories in the CIFAR-10 dataset, and which training techniques most improve generalisation?"**

Image classification is a fundamental task in deep learning. The CIFAR-10 dataset is one of the most widely used benchmarks for evaluating CNN architectures. This project implements a custom CNN with intermediate blocks and an output block, then systematically assesses the impact of training techniques such as data augmentation, normalisation, dropout, weight decay, and learning rate scheduling on model performance.

### Chosen dataset

The dataset used was the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), loaded via `torchvision`. It contains 50,000 training samples and 10,000 test samples across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32x32 pixels with 3 colour channels.

### Chosen deep learning architecture

| Component | Description |
|---|---|
| **Custom CNN** | A stem convolution (3→64 channels) followed by 3 intermediate blocks and a custom output block |
| **Intermediate Block** | K=3 blocks, each with L=3 convolutional layers using BatchNorm2d and LeakyReLU, with a softmax-weighted combination of layer outputs |
| **Output Block** | Global average pooling followed by a fully connected layer with dropout (0.3) |

## Project Objectives 🎯

1. Load and preprocess the CIFAR-10 dataset with appropriate augmentation and normalisation.
2. Implement a custom CNN architecture with intermediate and output blocks.
3. Train the model using cross-entropy loss and the Adam optimiser with cosine annealing.
4. Evaluate performance using accuracy, confusion matrices, and per-class accuracy.
5. Compare three experiments to assess the impact of training techniques and training length.

## Environment 👩🏻‍💻

<p align="center">
  <img src="https://img.shields.io/badge/jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Google Colab"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" alt="GitHub"/>
</p>

## Stack 🛠️

<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/torchvision-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="torchvision"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/seaborn-4C72B0?style=flat&logo=seaborn&logoColor=white" alt="seaborn"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
</p>

## Repository Structure 🌲

```
.
├── .gitattributes
├── Project.ipynb
└── README.md
```

## Method 🧪

| Stage | What I did |
|---|---|
| **Data loading** | Loaded CIFAR-10 via `torchvision` (50,000 train / 10,000 test), batch size 64 |
| **Preprocessing** | Normalised with per-channel mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616) |
| **Augmentation** | Random horizontal flip (50% probability) and random crop with padding=4 on training set only |
| **Architecture** | Custom CNN: stem conv (3→64), 3 intermediate blocks (L=3 conv layers each), output block with dropout 0.3 |
| **Training** | Cross-entropy loss, Adam optimiser (lr=0.001, weight_decay=1e-4), cosine annealing LR scheduler, 50 epochs |
| **Evaluation** | Training/testing accuracy per epoch, batch loss curve, confusion matrix, per-class accuracy |

## Results 📊

| Experiment | Epochs | Train Accuracy | Test Accuracy | Key Techniques |
|---|---|---|---|---|
| **Experiment 1 (Baseline)** | 15 | 82.04% | 64.84% | Basic architecture, no improvements |
| **Experiment 2 (Improved)** | 30 | 80.38% | 81.12% | Augmentation, normalisation, dropout, weight decay, LR scheduler |
| **Experiment 3 (Extended)** | 50 | 85.01% | 84.09% | All techniques + extended training |

### Per-Class Accuracy (Experiment 3, 50 epochs)

| Class | Accuracy |
|---|---|
| airplane | 87.6% |
| automobile | 93.1% |
| bird | 78.7% |
| cat | 69.2% |
| deer | 83.0% |
| dog | 76.9% |
| frog | 87.7% |
| horse | 83.2% |
| ship | 91.5% |
| truck | 90.0% |

- **Experiment 3 achieved the highest test accuracy (84.09%)**, benefiting from both training techniques and extended training.
- The baseline model showed clear signs of overfitting, with test accuracy fluctuating significantly.
- Adding normalisation, augmentation, dropout, and weight decay in Experiment 2 reduced overfitting and improved test accuracy by ~16 percentage points.
- Extending training to 50 epochs allowed the cosine annealing scheduler to fully decay, further improving convergence.

## Key Takeaways 🔑

- Data augmentation and normalisation significantly reduce overfitting and improve generalisation on CIFAR-10.
- Dropout (0.3) and L2 weight decay (1e-4) act as effective regularisers within the output block.
- Cosine annealing benefits from longer training schedules, with 50 epochs outperforming 30 epochs.

## Recommendations for Improvements 📈

- **Hyperparameter tuning** of learning rate, weight decay, and dropout rate using grid or random search.
- **Deeper architectures** such as ResNet-style skip connections for improved gradient flow.
- **Additional augmentation** such as Cutout or Mixup to further regularise the model.
- **Cross-validation** for a more reliable estimate of model performance.
- **Class-specific improvements** for underperforming classes like cat (69.2%) and dog (76.9%).

## Reflection 🪞

This project deepened my understanding of the full deep learning workflow — from designing a custom CNN architecture to systematically evaluating training techniques through three experiments. The progression from a 64.84% baseline to 84.09% accuracy demonstrated how normalisation, augmentation, dropout, weight decay, and learning rate scheduling each contribute to better generalisation. Next, I want to explore deeper architectures with residual connections and advanced augmentation strategies.

## References 📚

- Chauhan, R., Ghanshala, K.K. and Joshi, R.C., 2018. Convolutional neural network (CNN) for image detection and recognition. *2018 First International Conference on Secure Cyber Computing and Communication (ICSCCC)*, pp. 278-282. IEEE.
- DeVries, T. and Taylor, G.W., 2017. Improved regularization of convolutional neural networks with cutout. *arXiv preprint arXiv:1708.04552*.
- He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 770-778.
- Krizhevsky, A. and Hinton, G., 2009. Learning multiple layers of features from tiny images. *University of Toronto*.
- Loshchilov, I. and Hutter, F., 2016. SGDR: Stochastic gradient descent with warm restarts. *arXiv preprint arXiv:1608.03983*.
- Salehin, I. and Kang, D.K., 2023. A review on dropout regularization approaches for deep neural networks. *Electronics*, 12(14), p. 3106.
- Shafiq, M. and Gu, Z., 2022. Deep residual learning for image recognition: A survey. *Applied Sciences*, 12(18), p. 8972.
- Thanapol, P. et al., 2020. Reducing overfitting and improving generalization in training CNN under limited sample sizes. *2020-5th International Conference on Information Technology (InCIT)*, pp. 300-305. IEEE.
