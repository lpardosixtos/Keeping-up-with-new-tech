# Text-to-image Generation

We decided to approach the text-to-image generation problem using Generative Adeversarial Networks, that have already showed amazing performance at generating images and can be applied to different tasks, such as atural image generation, image colorization, style transfer, and super-resolution. Lets start by reviewing the fundamentals of GAN.

## Generative Adversarial Networks

We can get an initial grasp of Generative Adversarial Networks' underlying idea by thinking of them as a game with two players: the generator and the discriminator. The generator tries to create the most realistic samples, while the discriminator's goal is to correctly classify samples as real or fake. 

Let $G$ denote the generator and $D$ the discriminator. The method is formalized by the following optimization problem

$$\min_G\max_D V(D,G) = \mathbb{E}_{\mathbf{x}\sim p_{data}(\mathbf{x})}[\log(D(\mathbf{x})]+\mathbb{E}_{\mathbf{z}\sim p_{z}(\mathbf{z})}[1-\log(D(G(\mathbf{z}))].$$

In practice, the generator and discriminator are both parametrized as neural networks, and the optimization problem will be solved by alternating optimization steps for 2 different tasks:

1. Generator step: Generate fake samples and use them to tune the parameters of $G$ in order to deceive the frozen classifier $D$.
2. Discriminator step: Use both, real and fake samples to train $D$ as a classifier.

Observe that one task is in charge of optimizing $G$ while the other takes care of only $D$. Once the the generator and the discriminator have been jointly trained, we cn use the generator alone to sample images from a noise vector. 

We suggest that you visit this [DCGAN Pytorch Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) for a deeper explanation and som well written code. You can even run their code in [Google Colab!](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/e9c8374ecc202120dc94db26bf08a00f/dcgan_faces_tutorial.ipynb).

## Conditioning GANs