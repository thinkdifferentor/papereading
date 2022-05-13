<!-- TOC -->

- [00 Attention Is All You Need](#00-attention-is-all-you-need)

<!-- /TOC -->


## [00 Attention Is All You Need](./Attention%20Is%20All%20You%20Need.pdf)
- Vaswani A, Shazeer N, Parmar N, et al./2017/NIPS/41767
- We propose a **new simple** network architecture, the Transformer, based solely on **attention mechanisms**, which can be described as mapping a **query** and a set of **key-value** pairs to an output, dispensing with recurrence and convolutions entirely.
- ![Transformer 1](./images/Transformer_1.png)
- ![Transformer 2](./images/Transformer_2.png)
- To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension **d_model = 512**. This is not same with the design of CNN (wide & shallow -> narrow & deep). 
- This **masking**, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions **less than i**. (**Multi-Head Attention with mask** )
- ![Transformer 3](./images/Transformer_3.png)
- We suspect that for **large values of d_k**, the dot products grow large in magnitude, pushing the softmax function into regions where it has **extremely small gradients**. To counteract this effect, we scale the dot products by 1/sqrt(d_k). So the attention is called **Scaled Dot-Product Attention**.
- Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. We can use **different attention function**(distence function) to compute the weight between the query with every key. (It's same to multi channel in CNN)
- ![Transformer 4](./images/Transformer_4.png)