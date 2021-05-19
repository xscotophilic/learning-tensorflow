# Natural Language Processing (NLP)

## Code Outline (Note: First go through all the theory):

<p align="center">
  <!-- <img src=".png" alt="Outline"/> -->
</p>

---

<p align="center">
  <!-- <img src=".png" alt="Preprocessing"/> -->
</p>

### Text Preprocessing

A big part of the preprocessing is something encoding. This entails representing each piece of data in a way that a computer can comprehend, hence the name encode.

- There’s many different ways of encoding such as Label Encoding, One Hot Encoding, and Word Embedding, which is designed specifically for embedding words (NLP).

- Label Encoding

  - Assume we're dealing with categorical data, such as cats and dogs. By assigning a number to each category, the computer will know how to represent/process them.

  - Good for categorical data, Not good for NLP.

- One Hot Encoding

  - It is a method for converting categorical data to integers or a vector of ones and zeros. The number of expected classes or categories determines the length of the vector.

  - One-Hot Encoding is a broad method for vectorizing any categorical feature. To create and update the vectorization, simply add a new entry in the vector with a one for each new category. However, because of the speed and simplicity.

  - <img src="ohe.png"/>

  - Problems with One-Hot Encoding:

    - The "curse of dimensionality" is introduced by adding a new dimension for each category.

    - The second issue is that it is difficult to extract meanings. Each word is embedded in its own dimension, and each word contains a single one and N zeros, where N is the number of dimensions.

- Word Embedding

  - Embedding is a method that requires a large amount of data, both in total and in repeated occurrences of individual exemplars, as well as a long training time. As a result, a dense vector with a fixed, arbitrary number of dimensions is produced.

  - When we try to visualise One Hot Encoding for a sentence "Her face a river", we can imagine a four-dimensional space in which each word occupies one dimension and has nothing to do with the others.

  - Our goal is for words with similar contexts to occupy close spatial positions. This word embedding model reduces dimensionality while retaining information on contextual similarity.

  - Word2Vec is one method for creating such an embedding. It can be obtained through two methods (both of which involve Neural Networks): Skip Gram and Common Bag Of Words (CBOW).

#### Which method to choose?

Embeddings are a good choice if you have enough training data, enough training time, and the ability to use a more complex training algorithm (e.g., word2vec or GloVe). Otherwise, revert to One-Hot Encoding.

- You can read about Text Preprocessing in details on provided links:

  - [Introduction to Word Embedding and Word2Vec on TDS](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)

  - [Word Embedding in NLP: One-Hot Encoding and Skip-Gram Neural Network on TDS](https://towardsdatascience.com/word-embedding-in-nlp-one-hot-encoding-and-skip-gram-neural-network-81b424da58f2)

---
