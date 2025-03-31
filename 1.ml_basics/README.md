# ML basics

- Outline
  1. Classification (classification.ipynb)
  2. Regression (regression.ipynb)
  3. Neurons ([theory](#neurons-in-ai))

## Neurons in AI

Imagine your brain is like a giant network of tiny messengers called neurons. These neurons help you think, learn, and make decisions. In Artificial Intelligence (AI), we try to mimic how the brain works using artificial neurons. These artificial neurons are the building blocks of neural networks, which help computers learn from data and make smart decisions.

### How an Artificial Neuron Works
Think of an artificial neuron like a detective solving a mystery. It gathers clues (inputs), weighs their importance (weights), and then makes a final decision based on the gathered evidence (activation function).

### Parts of an Artificial Neuron

- **Inputs (x₁, x₂, ... xₙ):** Raw information the neuron receives for decision-making. (Like a detective gathering clues—color, shape, or size when identifying an animal.)
- **Weights (w₁, w₂, ... wₙ):** Each input is assigned a numerical weight, indicating its importance. Higher weights mean greater influence on the neuron's decision. (Like a detective ranking clues by importance—a fingerprint might be more critical than a witness statement.)
- **Bias (b):** A constant value added to the weighted sum to fine-tune decisions, ensuring the neuron doesn’t rely solely on inputs. (Like a detective’s intuition—a gut feeling that pushes the case one way or another even when clues are weak.)
- **Summation Function:** Computes the weighted sum of inputs and adds bias (Like a detective piecing together all the clues, weighing each one carefully, and forming an overall judgment about the case.)
  - `z = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + b`
- **Activation Function:** Determines whether the neuron **"fires"** (activates) or remains **"inactive"**.
  - **Sigmoid:** Produces values between **0 and 1**, meaning the neuron is either barely active or fully active. (Like a dimmer switch, where weak clues result in a low confidence level, while strong evidence lights up the case.)
  - **ReLU (Rectified Linear Unit):** If the input is **positive, it stays the same**. If negative, it becomes **zero**. (Either a clue is useful (positive) and helps solve the case, or it’s ignored (zero) as irrelevant.)
  - **Tanh:** Similar to sigmoid but ranges from **-1 to 1**, meaning it considers both positive and negative signals. (Like a detective weighing both supporting and contradicting evidence to make a more balanced judgment.)
  - **Softmax:** Converts multiple raw scores into **probabilities** that sum to **1**, helping AI choose between multiple options. (If the detective has multiple suspects, Softmax assigns probabilities: "Cat: 70%, Dog: 30%" and picks the most likely culprit.)

### Types of Artificial Neurons

1. **Perceptron:** A simple neuron that makes binary (yes/no) decisions based on weighted inputs. (Like a rookie detective solving basic cases—either the suspect is guilty or not, based on the available evidence.)
2. **Multilayer Perceptron (MLP):** A network of multiple perceptrons arranged in layers, allowing the model to learn complex patterns. (Like a team of detectives working together—each detective specializes in a different part of the case, passing insights to the next until a final conclusion is reached.)
3. **Convolutional Neurons:** Neurons designed for image recognition, identifying patterns such as edges, shapes, and textures. (Like forensic specialists—detectives who focus on crime scene evidence, such as fingerprints or footprints, to uncover important details.)
4. **Recurrent Neurons:** Neurons that remember past inputs, making them useful for analyzing sequential data like speech or time-series predictions. (Like an experienced detective who recalls past cases and patterns to solve ongoing investigations—using memory to connect the dots.)

### Role of Neurons in Neural Networks

- **Input Layer:** Neurons receive raw data, processing the initial information. (Like a detective gathering all clues and testimonies before starting the investigation.)
- **Hidden Layers:** Neurons work together to refine, transform, and extract patterns from the data. (Like a team of detectives analyzing clues, cross-referencing facts, and eliminating false leads to uncover deeper connections.)
- **Output Layer:** The final layer produces the network’s decision or prediction. (Like the lead detective making the final verdict—identifying the suspect, solving the case, or presenting a well-reasoned conclusion.)

### How AI Neurons Learn

AI neurons learn from mistakes:

1. **Loss Function:** Measures how far off the AI’s answer was. (Did the detective accuse the wrong suspect? If so, how badly was the mistake?)
2. **Gradient Descent:** Adjusts the neuron’s thinking process to **improve accuracy over time**. (The detective learns from errors, correcting their judgment on similar cases in the future.)
3. **Backpropagation:** The AI retraces its steps, fixing mistakes by **adjusting the weights of clues** in the network. (Like a detective going back over their notes, realizing which clues were misleading, and correcting their case report.)
4. **Optimization Algorithms:** Techniques like **Adam, RMSprop, or Stochastic Gradient Descent (SGD)** help AI **learn more efficiently**. (Like giving detectives better training methods to crack cases faster and more accurately.)

### Real-World Applications
Artificial neurons are at work solving all kinds of real-world mysteries:
- **Image Recognition:** Spotting faces, animals, and objects in photos.
- **Natural Language Processing (NLP):** Chatbots, translation apps, and AI that understands what you type.
- **Speech Recognition:** Turning spoken words into text, like Siri or Google Assistant.
- **Autonomous Driving:** AI "detectives" analyzing the road, making split-second driving decisions.
- **Medical Diagnosis:** Helping doctors identify diseases in X-rays and scans.
