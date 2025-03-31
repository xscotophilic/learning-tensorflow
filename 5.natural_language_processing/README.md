# Natural Language Processing (NLP)

## Code Outline (Note: First go through all the theory):

| Sequence No. | Notebook Name | Description |
|-------------|--------------|-------------|
| 1 | `text_preprocessing.ipynb` | Notebook for preprocessing text data for NLP tasks. |
| 2 | `spam_detection_using_lstm.ipynb` | Spam detection using a Long Short-Term Memory (LSTM) neural network. |
| 3 | `spam_detection_using_cnn.ipynb` | Spam detection using a Convolutional Neural Network (CNN). |

## Text Preprocessing

A major aspect of text preprocessing is encoding. This involves representing data in a format that a computer can process, which is known as encoding.

- There are many different encoding methods, such as Label Encoding, One-Hot Encoding, and Word Embedding, with Word Embedding being specifically designed for Natural Language Processing (NLP).

### Label Encoding

- Suppose we are dealing with categorical data, such as "cats" and "dogs." By assigning a numerical value to each category, the computer can effectively process them.
- **Suitable for categorical data but not ideal for NLP.**

### One-Hot Encoding

- One-Hot Encoding converts categorical data into integer vectors, where each category is represented as a binary vector of ones and zeros. The length of the vector is determined by the number of unique categories.
- This method is commonly used for vectorizing categorical features. To create and update the vector representation, a new entry is added with a '1' for each new category. However, despite its speed and simplicity, One-Hot Encoding has some limitations.
  - <img src="https://user-images.githubusercontent.com/47301282/118823580-b4288a80-b8d6-11eb-81e9-4efd5b3441fe.png"/>

#### Problems with One-Hot Encoding:

- **Curse of Dimensionality:** Each unique category adds a new dimension, significantly increasing the data's complexity.
- **Lack of Semantic Meaning:** Each word is embedded in its own dimension, containing a single '1' and multiple '0's, making it difficult to capture relationships between words.

### Word Embedding

- Word Embedding requires a large dataset and substantial training time. It transforms words into dense vectors with fixed dimensions while retaining contextual relationships.
- For example, when visualizing One-Hot Encoding for the sentence "Her face is a river," we can imagine a multi-dimensional space where each word occupies a separate dimension. However, One-Hot Encoding does not capture contextual similarity.
- The goal of Word Embedding is to ensure that words with similar meanings are positioned closely in a multi-dimensional space, improving NLP performance.
- **Word2Vec** is a widely used method for generating word embeddings. It has two primary approaches:
  - **Skip-Gram**
  - **Continuous Bag of Words (CBOW)**

### **Which Encoding Method to Choose?**

If you have sufficient training data, computational resources, and time, Word Embeddings (e.g., Word2Vec, GloVe) are a superior choice for NLP tasks. Otherwise, One-Hot Encoding remains a simpler alternative.

### Additional Reading:
- [Introduction to Word Embedding and Word2Vec on TDS](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)
- [Word Embedding in NLP: One-Hot Encoding and Skip-Gram Neural Network on TDS](https://towardsdatascience.com/word-embedding-in-nlp-one-hot-encoding-and-skip-gram-neural-network-81b424da58f2)

## Text Classification

According to Google: *Text classification algorithms are at the heart of a variety of software systems that process text data at scale. Email software uses text classification to determine whether incoming mail is sent to the inbox or filtered into the spam folder. Discussion forums use text classification to determine whether comments should be flagged as inappropriate.*

<img src="https://user-images.githubusercontent.com/47301282/119089291-8bfc7100-ba27-11eb-84c8-9b8df8ac3976.png" alt="workflow"/>

(Image by Google)

Different models can be used for text classification, including:
- **Traditional Machine Learning Models:** Multinomial Naïve Bayes, Support Vector Machines (SVM), Logistic Regression, Boosting.
- **Deep Learning Models:** Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN).

### Additional Reading:
- [Text Classification Guide by Google](https://developers.google.com/machine-learning/guides/text-classification)

## Model Selection in Text Classification

**Which model should you choose?** The decision can be unclear when considering multiple factors.

Traditional methods like **Naïve Bayes, SVM, Logistic Regression, and Boosting** work well for structured text data, while deep learning models like **LSTM, CNN, and RNN** are better suited for complex NLP tasks with large datasets.

Experimentation with different models is essential to determine the best fit for your specific use case.
