# Deep Learning with PyTorch 1.X

Implementations of neural network models with torch (>=1.0)

See also implementations with TesorFlow 2.0 [here](https://github.com/yusugomori/deeplearning-tf2).

## Requirements

* PyTorch >= 1.0

```shell
$ pip install torch torchvision
```

## Models

* Logistic Regression
* MLP
* LeNet
* ResNet (ResNet34, ResNet50)
* Encoder-Decoder (LSTM)
* Encoder-Decoder (Attention)
* Transformer
* Deep Q-Network
* Generative Adversarial Network
* Conditional GAN

```
models/
├── conditonal_gan_mnist.py
├── dqn_cartpole.py
├── encoder_decoder_attention.py
├── encoder_decoder_lstm.py
├── gan_fashion_mnist.py
├── lenet_mnist.py
├── logistic_regression_mnist.py
├── mlp_mnist.py
├── resnet34_fashion_mnist.py
├── resnet50_fashion_mnist.py
├── transformer.py
│
└── layers/
    ├── Attention.py
    ├── DotProductAttention.py
    ├── Flatten.py
    ├── GlobalAvgPool2d.py
    ├── MultiHeadAttention.py
    ├── PositionalEncoding.py
    └── ScaledDotProductAttention.py
```
