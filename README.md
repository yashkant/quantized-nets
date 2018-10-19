# Quantized-Nets

**This mini-project contains code for building [Binary][1], [Ternary][2] and [N-bit Quantized][3] Convolutional Neural Networks built with Keras.** 


Introduction
------------
Low Precision Networks have recently gained popularity due to their applications in devices with low-compute capabilities. During the forward pass, QNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations. 

Various Binarization, Ternarization and Quantization schemes are published for weights and activations. 


<p align="center">
  <img src="https://imgur.com/ASAVPwX.png">
</p>


Image Source: [Minimum Energy Quantized Neural Networks][6]

Binarization function used in the experiment is deterministic binary-tanh which is placed in [```binary_ops.py```][3]


Setup Dependencies
-----
The recommended version for running the experiments is Python3.

1. Follow the installation guide on [Tensorflow Homepage][4] for installing Tensorflow-GPU or Tensorflow-CPU. 
2. Follow instructions outlined on [Keras Homepage][5] for installing Keras.


Project Structure
-----------------
The skeletal overview of the project is as follows: 

```bash
.
├── binarize/
│   ├── binary_layers.py  # Custom binary layers are defined in Keras 
│   └── binary_ops.py     # Binarization functions for weights and activations
|
├── ternarize/
│   ├── ternary_layers.py  # Custom ternarized layers are defined in Keras
│   └── ternary_ops.py     # Ternarization functions for weights and activations
|
├── quantize/
│   ├── quantized_layers.py  # Custom quantized layers are defined in Keras
│   └── quantized_ops.py     # Quantization functions for weights and activations
|
├── base_ops.py           # Stores generic operations              
├── binary_net.py         # Implementation of Binarized Neural Networks
├── ternary_net.py        # Implementation of Ternarized Neural Networks
└── quantized_net.py      # Implementation of Quantized Neural Networks
```

Usage
----------
In the root directory, to run the examples use: 

```bash 
python {example}_net.py
```

Also, you can import the layers directly in your own Keras or Tensorflow code. Read [this blog][7] to know how to use Keras layers in Tensorflow


[1]:https://arxiv.org/abs/1602.02830
[2]:https://arxiv.org/abs/1605.04711
[3]:https://arxiv.org/abs/1609.07061
[4]:https://www.tensorflow.org/install/
[5]:https://keras.io/#installation
[6]:https://arxiv.org/pdf/1711.00215.pdf
[7]:https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html



