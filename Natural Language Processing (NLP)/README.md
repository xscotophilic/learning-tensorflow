# Natural Language Processing (NLP)

## Code Outline (Note: First go through all the theory):

<p align="center">
  <img src="https://user-images.githubusercontent.com/47301282/119091939-6b361a80-ba2b-11eb-8c63-0ece6e8dd2c9.png" alt="Outline"/>
</p>

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/47301282/118823459-9fe48d80-b8d6-11eb-9ce8-a23361701f6b.png" alt="Preprocessing"/>
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

  - <img src="https://user-images.githubusercontent.com/47301282/118823580-b4288a80-b8d6-11eb-81e9-4efd5b3441fe.png"/>

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

<p align="center">
  <img src="https://user-images.githubusercontent.com/47301282/119089235-75eeb080-ba27-11eb-98cd-c8c40a3b0d7a.png" alt="TextClassification"/>
</p>

### Text Classification

Google says: Text classification algorithms are at the heart of a variety of software systems that process text data at scale. Email software uses text classification to determine whether incoming mail is sent to the inbox or filtered into the spam folder. Discussion forums use text classification to determine whether comments should be flagged as inappropriate.

<img src="https://user-images.githubusercontent.com/47301282/119089291-8bfc7100-ba27-11eb-84c8-9b8df8ac3976.png" alt="workflow"/>

Image by Google

There are different models (eg. LSTM, CNN, multinomial Naïve Bayes, SVM, Logistic Regression, boosting) which you can try and decide which is best fit for you.

- You can read about Text Preprocessing in details on provided links:

  - [Text-classification-guide by Google](https://developers.google.com/machine-learning/guides/text-classification)

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/47301282/119089343-9c145080-ba27-11eb-9e1b-a1ad19888d20.png" alt="ModelSelection"/>
</p>

### Model Selection in Text Classification

Which model should I go with? When considering various aspects, it becomes hazy. I'm talking about how to compare traditional methods (multinomial Naïve Bayes, SVM, Logistic Regression, boosting, etc.) and neural networks (LSTM, RNN, CNN, etc.). Various models should be tried to determine which is the best fit for you.
