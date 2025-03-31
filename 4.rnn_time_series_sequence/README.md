# Recurrent Neural Networks, Time Series, and Sequence Data

## Code Outline (Note: First go through all the theory)

| Sequence No. | Notebook Name | Description |
|-------------|--------------|-------------|
| 1 | `autoregressive_model.ipynb` | An autoregressive (AR) model predicts future behavior based on past data. |
| 2 | `simple_rnn_sine.ipynb` | A simple RNN network predicting a sine wave. |
| 3 | `rnn_shapes.ipynb` | Understanding input, output, and hidden layer shapes in RNNs. More details in [Time Series Theory](#time-series). |
| 4 | `lstm_nonlinear.ipynb` | Experimenting with LSTM for nonlinear sequence modeling. |
| 5 | `lstm_long_distance.ipynb` | Exploring how LSTMs handle long-distance dependencies better than RNNs, overcoming the vanishing gradient problem. |
| 6 | `rnn_image_classification.ipynb` | Using RNNs for image classification tasks. |
| 7 | `rnn_stock_returns_forecasting.ipynb` | Predicting future stock prices of Starbucks (SBUX) using a Recurrent Neural Network (RNN). |

## Sequence Data

A sequence imposes an explicit order on observations. This order is crucial and must be preserved when formulating prediction problems where sequence data is used as input or output for a model.

- **Sequence Prediction**
  - Predicting the next value in a given input sequence is the core of sequence prediction.
  - **Example:**
    - Given: 1, 2, 3, 4, 5
    - Predict: 6
- **Examples of Sequence Data:**
  - Protein sequences
  - Gene sequences

## Time Series

A time series is most commonly a sequence of observations recorded at successive, equally spaced points in time. As a result, it is a series of discrete-time data. Examples of time series data include ocean tide heights, sunspot counts, and more.

- Generally, a time series is represented as **N × D × T**, where:
  - **N** = Batch size
  - **D** = Number of features
  - **T** = Number of timesteps
- **Example:** Suppose we want to model the path **X** takes to get to the library.
  - One sample represents **X’s single trip to the library**.
  - **D = 2**: The GPS records (latitude, longitude) pairs.
  - **T**: The number of (latitude, longitude) measurements taken from start to finish of a single trip.
    - Example: If the trip lasts **30 minutes** and coordinates are recorded **every second**, then **T = 1800**.
  - A coding example can be found in `rnn_shapes.ipynb`.

### Characteristics of Time Series

1. **Autocorrelation**
   - Autocorrelation refers to the similarity between observations as a function of the time lag between them.
   - ![Autocorrelation](https://user-images.githubusercontent.com/47301282/118347622-468cff00-b562-11eb-9add-f5cdfd85da16.png) *Image source: towardsdatascience.com*
2. **Seasonality**
   - Seasonality refers to periodic fluctuations in a time series.
   - ![Seasonality](https://user-images.githubusercontent.com/47301282/118347632-5dcbec80-b562-11eb-833a-668e4f851573.png) *Image source: towardsdatascience.com*
   - If the autocorrelation plot exhibits a sinusoidal pattern, seasonality can be inferred. Simply examining the time axis can indicate the duration of the season.
3. **Stationarity**
   - A time series is considered *stationary* if its statistical properties remain constant over time. Specifically:
     - The **mean** and **variance** are constant.
     - The **covariance** is independent of time.
   - ![Stationarity](https://user-images.githubusercontent.com/47301282/118347647-7936f780-b562-11eb-9bda-b916a8ca4c91.png) *Image source: towardsdatascience.com*
   - Ideally, a stationary time series is preferable for modeling. However, real-world data is often non-stationary, and various transformations can be applied to achieve stationarity.

### Modeling Time Series

There are several methods for modeling a time series to make predictions. Below are some commonly used approaches:

1. **Moving Average (MA)**
   - This model predicts the next observation as the mean of all previous observations.
   - ![Moving Average](https://user-images.githubusercontent.com/47301282/118347670-966bc600-b562-11eb-8db4-5c5b55ff4f44.png) *Image source: towardsdatascience.com*
2. **Exponential Smoothing**
   - Similar to the moving average, but applies exponentially decreasing weights to past observations.
   - More recent observations are given greater importance, while older ones gradually become less relevant.
   - ![Exponential Smoothing](https://user-images.githubusercontent.com/47301282/118347685-b3a09480-b562-11eb-9543-46b27f233fb3.png) *Image source: towardsdatascience.com*
3. **Seasonal Autoregressive Integrated Moving Average (SARIMA)**
   - A dynamic model that accounts for non-stationarity and seasonality by combining simpler models.
   - **Components of SARIMA:**
     - **Autoregression (AR(p))**: A time series regression on its past values, where **p** represents the maximum lag.
       - Determined using the **Partial Autocorrelation Plot (PACF)**.
       - ![Partial Autocorrelation](https://user-images.githubusercontent.com/47301282/118347692-c5823780-b562-11eb-9357-3811b332b9c9.png) *Image source: towardsdatascience.com*
     - **Moving Average (MA(q))**: A model where the next observation depends on past errors, with **q** representing the maximum lag.
       - Determined using the **Autocorrelation Plot (ACF)**.
       - ![Autocorrelation](https://user-images.githubusercontent.com/47301282/118347710-d8950780-b562-11eb-94d2-4defee6a6f3c.png) *Image source: towardsdatascience.com*
     - **Integration (I(d))**: Represents the number of differences needed to achieve stationarity.
     - **Seasonality (S(P, D, Q, s))**: Models periodic fluctuations, where **s** is the seasonal length.
   - **Final Model:** SARIMA(**p, d, q**) × (**P, D, Q, s**)
     - The key takeaway is that before applying SARIMA, the time series must be transformed to remove seasonality and non-stationary behaviors.

### Additional Resources

For a deeper dive into time series analysis and modeling, refer to the following resources:

- 📖 [The Complete Guide to Time Series Analysis and Forecasting on TDS](https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775)
- 📖 [Recurrent Neural Networks Cheatsheet (Afshine & Shervine Amidi)](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

## Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a type of neural network that allows previous outputs to be used as inputs while maintaining hidden states.

The image below shows a simple neural network:

<img src="https://user-images.githubusercontent.com/47301282/118347715-ed719b00-b562-11eb-8653-c5782c500cf3.png"/>

*Image source: towardsdatascience.com*

The image below shows a recurrent neural network:

<img src="https://user-images.githubusercontent.com/47301282/118347720-fd897a80-b562-11eb-82c0-cf1a98db5e9f.png"/>

*Image source: towardsdatascience.com*

### Advantages of RNNs
- The ability to process inputs of any length.
- The model's size does not increase with the size of the input.
- Weights are shared over time.

### How RNNs Learn
- RNNs remember past inputs and base their decisions on prior information. Basic feed-forward networks also retain information but only what they learn during training.
- While RNNs learn during training, they also retain knowledge from previous inputs while generating outputs. This enables them to produce different outputs for the same input depending on the prior sequence.
- RNNs can take one or more input vectors and generate one or more output vectors. Unlike typical neural networks, the output is influenced not only by weights applied to inputs but also by a "hidden" state vector representing prior inputs/outputs.

### Deep RNNs
- Can an RNN be made "deep" to achieve the multi-level abstractions gained by depth in typical neural networks?
- Four methods to add depth:
  1. Stack multiple hidden states, feeding the output of one to the next.
  2. Introduce nonlinear hidden layers between input and hidden states.
  3. Deepen the hidden-to-hidden transition. [This paper](https://arxiv.org/pdf/1312.6026.pdf) by Pascanu et al. explores deep RNNs and demonstrates their advantages over shallow RNNs.

### Bidirectional RNNs
- Learning from the past alone is insufficient; looking into the future can help correct the past.
  - <img src="https://user-images.githubusercontent.com/47301282/118347728-0f6b1d80-b563-11eb-9c2b-68d080d35bdd.png"/> *Image source: towardsdatascience.com*
- However, this raises the question: how far into the future should we look? If we wait for all inputs, the operation becomes computationally expensive.

### Recursive Neural Networks
- RNNs process inputs sequentially. Recursive Neural Networks (RecNNs) apply transitions repeatedly but not necessarily in sequence.
  - <img src="https://user-images.githubusercontent.com/47301282/118347735-1e51d000-b563-11eb-8ac9-c8c71f9f9bd7.png"/> *Image source: towardsdatascience.com*
- **Recursive Neural Networks are a subset of Recurrent Neural Networks.** They can operate on hierarchical tree structures by parsing input nodes, merging child nodes into parent nodes, and combining them with other nodes. RNNs perform the same function but follow a purely linear structure.
- In RNNs, weights are applied sequentially to each input node. This structured approach allows conventional training methods like backpropagation. However, if the structure is not predetermined, is it also learned?

### RNN Architectures

<img src="https://user-images.githubusercontent.com/47301282/118347743-2c075580-b563-11eb-8388-fa9e2fadfb8b.png"/>

1. **One-to-Many Architecture**: Example – Image captioning. A single image input generates a sequence of descriptive words.
2. **Many-to-One Architecture**: Example – Sentiment analysis. A sequence of words (sentence) is classified as positive or negative.
3. **Many-to-Many Architecture**:
  - **Type 1**: Input length equals output length.
  - **Type 2**: Input length differs from output length.

### Encoder-Decoder Sequence-to-Sequence RNNs
- Encoder-Decoder or Sequence-to-Sequence RNNs are widely used in translation services.
- Two RNNs are used:
  - **Encoder**: Continuously updates its hidden state and produces a single "context" output.
  - **Decoder**: Translates the context into a sequence of outputs.
- A key advantage is that input and output sequence lengths do not have to match.
  - <img src="https://user-images.githubusercontent.com/47301282/118347751-39bcdb00-b563-11eb-8d06-e157a66e60cc.png"/> *Image source: towardsdatascience.com*

## LSTM & GRU

Recurrent Neural Networks (RNNs) suffer from short-term memory. They struggle to retain information from earlier time steps when processing long sequences. For example, if you're trying to predict something based on a paragraph of text, RNNs might fail to retain important information from the beginning.

LSTMs and GRUs were developed as a solution to short-term memory loss. They incorporate internal mechanisms called gates that regulate the flow of information.

<img src="https://user-images.githubusercontent.com/47301282/118347759-480af700-b563-11eb-9d41-d20d7a3644b7.png"/>

(LSTM and GRU) Image source: towardsdatascience.com

- These gates determine which data in a sequence should be retained or discarded. They allow relevant information to be transferred across long sequences, leading to improved predictions. These two architectures power nearly all state-of-the-art recurrent neural network models.

### LSTMs
- Traditional RNNs are ineffective at capturing long-term dependencies, primarily due to the **vanishing gradient problem**. When training deep networks, gradients (or derivatives) diminish exponentially as they propagate through layers. This prevents neural networks from updating their weights effectively, sometimes completely halting learning. The vanishing gradient problem is particularly common in deep neural networks.
- **Long Short-Term Memory (LSTM)** networks were developed by Sepp Hochreiter and Juergen Schmidhuber to address this issue. Unlike traditional RNNs, LSTMs include a specialized hidden layer that enables them to retain information for extended periods. In addition to the hidden state, an **LSTM cell** introduces a **cell state**, which is passed to the next time step.
  - <img src="https://user-images.githubusercontent.com/47301282/118347766-5822d680-b563-11eb-9dcf-3075b532d805.png"/>
- LSTMs are designed to capture long-term dependencies. They include three primary gates that regulate memory retention and update processes:
  - **Forget Gate**: Determines which information should be discarded from the cell state.
  - **Input Gate**: Adds new, relevant information to the cell state.
  - **Output Gate**: Determines what information from the cell state should be sent as output.
- [This post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) provides an excellent introduction to LSTMs. In a vanilla RNN, the input and the hidden state pass through a single tanh layer. **LSTMs enhance this mechanism by introducing additional gates and a cell state**, solving the issue of maintaining or resetting context over sequences. **GRUs** (Gated Recurrent Units) are a variant of LSTMs that use gates differently to address long-term dependencies.

### Additional Reading
- **RNNs:**
  - [Recurrent Neural Networks, E. Scornet](https://erwanscornet.github.io/teaching/RNN.pdf)
  - [Recurrent Neural Networks on TDS](https://towardsdatascience.com/recurrent-neural-networks-d4642c9bc7ce)
  - [Recurrent Neural Networks Cheatsheet - Afshine Amidi & Shervine Amidi](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)
- **LSTM & GRU:**
  - [Illustrated Guide to LSTMs and GRUs: A Step-by-Step Explanation on TDS](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
