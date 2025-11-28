# ANN-From-Scratch-
"DL models implemented from scratch using NumPy and Pandas only"
🧠 Artificial Neural Network (ANN) — From Scratch

Mini-Batch GD | Adam Optimizer | Dropout | Full Forward + Backprop | Visualizations

This repository contains a complete scratch implementation of a Deep Neural Network (DNN) trained on a synthetic dataset.
Everything is implemented from first principles—no TensorFlow, no PyTorch.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 What This Project Covers

Forward propagation

Backpropagation

Parameter initialization

Activation functions

Mini-Batch Gradient Descent

Adam Optimizer

Dropout Regularization

Loss computation

Gradient updates

Training loop

Visualizations to build intuition

Full mathematical explanations



-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🏗️ 1. Architecture Overview

A general L-layer deep neural network:

$$a^{[0]}$$ = X

For each layer :

$$z^{[l]}$$ = $$W^{[l]} a^{[l-1]} + b^{[l]}$$

$$a^{[l]}$$= $$\text{Activation}(z^{[l]})$$

Final output depends on task:

Classification → sigmoid/softmax

Regression → linear



-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

⚙️ 2. Forward Propagation

Supported activations:

ReLU

TanH

Sigmoid


$$\text{ReLU}(z)$$=$$\max(0,z)$$

$$\tanh(z)$$=$$\frac{e^z - e^{-z}}{e^z+e^{-z}}$$


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔁 3. Backpropagation

Derived using chain rule:

Gradients

$$dZ^{[l]}$$ = $$dA^{[l]} \odot g'(Z^{[l]})$$

$$dW^{[l]}$4 = $$\frac{1}{m} dZ^{[l]} A^{[l-1]T}$$

$4db^{[l]}$$ = $$\frac{1}{m} \sum dZ^{[l]}$$

$$dA^{[l-1]}$$ = $$W^{[l]T} dZ^{[l]}$$


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📉 4. Loss Function

For binary classification:

$$\mathcal{L}$$ = $$-\frac{1}{m} \sum (y\log\hat{y} + (1-y)\log(1-\hat{y}))$$


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧩 5. Mini-Batch Gradient Descent

Dataset is split into batches of size batch_size:

X = $$[X^{(1)}, X^{(2)}, \dots]$$

Benefits:

Faster iterations

Smoother convergence

Less memory usage



-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 6. Adam Optimizer (From Scratch)

Adam mixes Momentum + RMSProp:

1. Update first moment:

$$v_t$$ = $$\beta_1 v_{t-1} + (1-\beta_1) g_t$$

2. Update second moment:

$$s_t$$ = $$\beta_2 s_{t-1} + (1-\beta_2) g_t^2$$

3. Bias correction:

$$\hat{v}_t$$ = $$\frac{v_t}{1-\beta_1^t}$$

$$\hat{s}_t$$ = $$\frac{s_t}{1-\beta_2^t} $$

4. Parameter update:

$$\theta$$ = $$\theta - \alpha \frac{\hat{v}_t}{\sqrt{\hat{s}_t}+\epsilon}$$


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧨 7. Dropout Regularization

To avoid overfitting:

$$D^{[l]}$$ = $$( \text{rand}(A^{[l]}) < keep\_prob )$$

$$A^{[l]}$$ = $$\frac{A^{[l]} \odot D^{[l]}}{keep\_prob}$$

During inference → dropout disabled.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📦 8. Caching System

During forward pass you store:

$$z^{[l]}$$

$$a^{[l]}$$

dropout masks

parameters


Cache is essential for computing backward propagation efficiently.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔍 9. Visualizations Included

This repo provides intuitive plots:

🎯 Loss curve , Acuuracy curve

Visualizing convergence.
🎯 Neural Network Layers as Feature Transformers (2D → 3D → 2D)

🎯 Activation functions

ReLU, TanH, Sigmoid plots.

🎯 Decision boundary

Shows how the neural network separates classes.

🎯 Layer activation

to show neuron firing patterns.

🎯 Dropout Mask Effect

🎯 Mini-Batch Gradient Descent Visualization

🎯 Adam Optimizer m & v Moment Visualization

🎯 Adam vs RMSProp vs SGD comparison plots

Understanding optimizer behavior.

🎯 Learning rate effect visualization

Training curves for different .

🎯 Backpropagation Gradient Flow (Vanishing/Exploding)

🎯 Training Loss Curve (Typical Deep Network)

🎯 Cross-Entropy Loss Surface Visualization

🎯 Weight Updates Over Iterations Visualization


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧪 10. Dataset

Synthetic Dataset


All features scaled before training.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 📁 12. Repository Structure
├── ann_scratch.ipynb
├── visualizations/
│   ├── loss_curve.png
│   ├── Neural Network Layers as Feature Transformers (2D → 3D → 2D).png
│   ├── decision_boundary.png
│   ├── optimizers_compare.png
│   ├── Dropout Mask Effect Visualization.png
└── README.md
