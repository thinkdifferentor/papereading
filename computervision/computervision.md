<!-- TOC -->

- [00 Generative Adversarial Nets](#00-generative-adversarial-nets)
- [01 Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey](#01-self-supervised-visual-feature-learning-with-deep-neural-networks-a-survey)
- [02 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](#02-an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale)
- [03 Swin Transformer Hierarchical Vision Transformer using Shifted Windows](#03-swin-transformer-hierarchical-vision-transformer-using-shifted-windows)
- [04 Learning Transferable Visual Models From Natural Language Supervision](#04-learning-transferable-visual-models-from-natural-language-supervision)
- [05 Masked Autoencoders Are Scalable Vision Learners](#05-masked-autoencoders-are-scalable-vision-learners)

<!-- /TOC -->

## [00 Generative Adversarial Nets](./Generative%20Adversarial%20Nets.pdf)
- Goodfellow I, Pouget-Abadie J, Mirza M, et al./NIPS/2014/44268
- We propose a new framework for estimating generative models via an **adversarial** process, in which we simultaneously train two models: a **generative model G** that captures the data distribution, and a **discriminative model D** that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to **maximize the probability of D making a mistake**.
- We train D to **maximize** the probability of assigning the correct label to both training examples and samples from G. We simultaneously train G to **minimize** log(1 − D(G(z))). In other words, D and G play the following **two-player minimax game** with value function V (G, D): ![GAN_1](./images/GAN_1.png)
- Optimizing D to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting. Instead, we alternate between k steps of optimizing D and one step of optimizing G. This results in D being maintained near its optimal solution, so long as G changes slowly enough. 
- ![GAN_2](./images/GAN_2.png)
- The hyperparameter *k* is important for training D model & G model. If the D model is weak, the G model will generate poor result and get the bad estimating of p_x. If the D model is strong, the update of G model will be very slow and we can't train a good G model. In a word, it's important to set a suitable value of *k* to **keep G model & D model training synchronously**.
- ![GAN_3](./images/GAN_3.png)
- In practice, equation 1 may not provide sufficient gradient for G to learn well. Early in learning, when G is poor, D can reject samples with high confidence because they are clearly different from the training data. In this case, log(1 − D(G(z))) saturates. Rather than training G to minimize log(1 − D(G(z))) we can train G to **maximize** log D(G(z)). This objective function results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning. (**New problem**: D(G(z)) might close to zero and log D(G(z)) close to infinit)

 
 ## [01 Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey](./Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks%20A%20Survey.pdf)
 - Jing L, Tian Y./2019/Pattern Analysis and Machine Intelligence/752.
 - ![Self-supervised Visual Feature Learning with Deep Neural Networks_1](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_1.png)
 - The pre-trained models and fine-tuned for other tasks for two main reasons: (1) the parameters learned from large-scale diverse datasets provide a good starting point, therefore, networks training on other tasks can **converge faster**, (2) the network trained on large-scale datasets already learned the hierarchy features which can help to **reduce over-fitting problem** during the training of other tasks, especially when datasets of other tasks are **small or training labels are scarce**.
- The pretext tasks share two common properties: (1) visual features of images or videos need to **be captured by ConvNets** to solve the pretext tasks, (2) pseudo labels for the pretext task can be **automatically generated** based on the attributes of images or videos. After the training on the pretext task is finished, **ConvNet models** that can capture visual **features** for images or videos are obtained.
- ![Self-supervised Visual Feature Learning with Deep Neural Networks_2](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_2.png)
- **Some Important Definition:**
    1. Supervised Learning: Supervised learning indicates learning methods using data with **fine-grained human-annotated labels** to train networks.
    2. Semi-supervised Learning: Semi-supervised learning refers to learning methods using a **small amount of labeled data** in conjunction with a **large amount of unlabeled data**.
    3. Weakly-supervised Learning: Weakly supervised learning refers to learning methods to learn with **coarse-grained labels or inaccurate labels**. The cost of obtaining weak supervision labels is generally **much cheaper** than fine-grained labels for supervised methods.
    4. **Unsupervised Learning**: Unsupervised learning refers to learning methods without using any human-annotated labels.(K-means, KNN)
    5. **Self-supervised Learning**: Self-supervised learning is a **subset of** unsupervised learning methods. Self-supervised learning refers to learning methods in which ConvNets are explicitly trained with **automatically generated labels**(Pseudo Lable).
- ![Self-supervised Visual Feature Learning with Deep Neural Networks_3](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_3.png)
- Generation-based Image Feature Learning:
    1. Image Generation with GAN: The discriminator is required to capture **the semantic features** from images to accomplish the discriminate task(**real data distribution or generated data distribution**). The **parameters of the discriminator** can server as the pre-trained model for other computer vision tasks. ![Self-supervised Visual Feature Learning with Deep Neural Networks_4](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_4.png)
    2. Image Generation with Inpainting:![Self-supervised Visual Feature Learning with Deep Neural Networks_5](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_5.png)
    3. Image Generation with Colorization:![Self-supervised Visual Feature Learning with Deep Neural Networks_6](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_6.png)
- Context-Based Image Feature Learning:
    1. Learning with Context Similarity:![Self-supervised Visual Feature Learning with Deep Neural Networks_7](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_7.png)
    2. Learning with Spatial Context Structure:![Self-supervised Visual Feature Learning with Deep Neural Networks_8](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_8.png)
- Free Semantic Label-based Image Feature Learning:
    1. Learning with Labels Generated by Game Engines: One problem needed to solve is how to bridge the domain gap between synthetic data and real-world data.![Self-supervised Visual Feature Learning with Deep Neural Networks_9](./images/Self-supervised%20Visual%20Feature%20Learning%20with%20Deep%20Neural%20Networks_9.png)
    2. Learning with Labels Generated by Hard-code programs: This type of methods generally has two steps: (1) label generation by employing hard-code programs on images or videos to obtain labels, (2) train ConvNets with the generated labels. Compared to other self-supervised learning methods, the supervision signal in these pretext tasks is **semantic labels(But very noisy)** which can directly drive the ConvNet to learn semantic features.


## [02 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](./An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.pdf)
- Dosovitskiy A, Beyer L, Kolesnikov A, et al./2021/ICLR/4852
- We show that this reliance on CNNs is not necessary and a **pure** transformer applied **directly** to sequences of image patches can perform **very** well on image classification tasks. Besides using patch to reduce the input number of Transformer, As an alternative to raw image patches, the input sequence can be formed from **feature maps** of a CNN.
- Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, **with the fewest possible modifications**. Image patches are treated the same way as tokens (words) in an NLP application.
- ![ViT_1](./images/ViT_1.png)
- ![ViT_2](./images/ViT_2.png)
- Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings.


## [03 Swin Transformer Hierarchical Vision Transformer using Shifted Windows](./Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows.pdf)
- Liu Z, Lin Y, Cao Y, et al./2021/ICCV/1752
- Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as **large variations in the scale** of visual entities and the **high resolution of pixels** in images compared to words in text. In this paper, we seek to expand the applicability of Transformer such that it can serve as a **general-purpose backbone** for computer vision, as it does for NLP and as CNNs do in vision.
- ![Swin_Transformer_1](./images/Swin_Transformer_1.png)
- A key design element of Swin Transformer is its shift of the window partition between consecutive self-attention layers.  The shifted windows bridge the windows of the preceding layer, **providing connections among them that significantly enhance modeling power**. This strategy is also efficient in regards to real-world latency: all query patches within a window share the same key set, which **facilitates memory access in hardware**.
- ![Swin_Transformer_2](./images/Swin_Transformer_2.png)
- ![Swin_Transformer_3](./images/Swin_Transformer_3.png)


## [04 Learning Transferable Visual Models From Natural Language Supervision](./Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf)
- Radford A, Kim J W, Hallacy C, et al./2021/PMLR/1279
- State-of-the-art computer vision systems are trained to predict a **fixed set of predetermined object categories**. This restricted form of supervision limits their **generality and usability** since **additional labeled data is needed to specify any other visual concept**. Learning directly from **raw text about images** (Image, Text) pair is a promising alternative which leverages a much broader source of supervision.
- Summary of **CLIP**. While standard image models jointly train an **image feature extractor** and a **linear classifier** to predict some label, CLIP jointly trains an **image encoder** and a **text encoder** to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes.
- ![CLIP_1](./images/CLIP_1.png)
- ![CLIP_2](./images/CLIP_2.png)
- ![CLIP_3](./images/CLIP_3.png)
- For the prediction task of a new label class, just add this new label to the label set and generate the corresponding text features through the text encoder. An image with a new category can generate the corresponding image feature through the trained image encoder, and then calculate the cosine similarity of all text features of this image feature, and then the category of the image can be output, although the image of this type has never appeared in the training set.


## [05 Masked Autoencoders Are Scalable Vision Learners](./Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.pdf)
- He K, Chen X, Xie S, et al. /2022/CVPR/261
- MAE approach is simple: **we mask random patches of the input image and reconstruct the missing pixels**. It is based on two core designs. First, we develop an **asymmetric** encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a **lightweight decoder** that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, yields a nontrivial and meaningful self-supervisory task.
- *what makes masked autoencoding different between vision and language?* : **1)** architectures were different; **2)** Information density is different between language and vision; **3)** The autoencoder’s decoder, which maps the latent representation back to the input, plays a different role between reconstructing text and images.
- ![MAE_1](./images/MAE_1.png)
- ![MAE_2](./images/MAE_2.png)
- ![MAE_3](./images/MAE_3.png)