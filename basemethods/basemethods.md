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