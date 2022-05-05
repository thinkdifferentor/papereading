# Paper Reading Notes of Base Methods

## [00 A Short Introduction to Learning to Rank](A%20Short%20Introduction%20to%20Learning%20to%20Rank.pdf)
- Hang LI./2011/IEICE/352
- Ranking is nothing but to select a **permutation** $π_i$ ∈ $Π_i$ for the given query qi and the associated documents $D_i$ using the scores given by the ranking model f(qi, di).
- Relevance judgments are usually conducted at five levels, for example, *perfect, excellent, good, fair, and bad*.
- **Ordinal classification** (also known as ordinal regression)：The goal of learning is to construct a model which can assign a grade label y to a given feature vector x. The model mainly consists of a scoring function f(x). In ranking, one cares more about accurate ordering of objects, while in ordinal classification, one cares more about accurate ordered-categorization of objects. 
- In the **pointwise** approach, the ranking problem (ranking creation) is transformed to classification, regression, or ordinal classification. The group structure of ranking is ignored in this approach.
- In the **pairwise** approach, ranking is transformed into pairwise classification or pairwise regression. In the formercase, a classifier for classifying the ranking orders of document pairs is created and is employed in the ranking of documents. In the pairwise approach, the group structure of ranking is also ignored.
- The **listwise** approach addresses the ranking problem in a more straightforward way. Specifically, it takes ranking lists as instances in both learning and prediction. The group structure of ranking is maintained and ranking evaluation measures can be more directly incorporated into the loss functions in learning.
- ![Model Architecture](./images/A%20Short%20Introduction%20to%20Learning%20to%20Rank.png)


## [01 Deep Metric Learning: A Survey](Deep%20Metric%20Learning%20A%20Survey.pdf)
- Mahmut KAYA, Hasan ¸Sakir B˙ILGE./2019/MDPI/180
- Metric learning is an approach based directly on a distance metric that aims to **establish similarity or dissimilarity between objects**. While metric learning aims to reduce the distance between similar objects, it also aims to increase the distance between dissimilar objects.
- ![Model Architecture1](./images/Deep%20Metric%20Learning_1.png)
- Deep metric learning consists of three main parts, which are **informative input samples**, **the structure of the network model**, and a **metric loss function**.  Informative samples are one of the most substantial elements that increase the success of deep metric learning.
- ![Model Architecture2](./images/Deep%20Metric%20Learning_2.png)
- ![Model Architecture3](./images/Deep%20Metric%20Learning_3.png)
- The Siamese network, as a metric learning approach, receives pair images, including positive and negative samples to train a network model. The distance between these pair images is calculated via a loss function.
- Triplet network inspired by Siamese network contains three objects, which are formed positive,negative and anchor samples. Triplet networks provide a higher discrimination power while using both in-class and inter-class relations.
- ![Model Architecture4](./images/Deep%20Metric%20Learning_4.png)
- ![Model Architecture4](./images/Deep%20Metric%20Learning_5.png)