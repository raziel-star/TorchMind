# TorchMind – Real-Time Natural Language Command Model

**TorchMind** is a lightweight, real-time neural network model built in PyTorch for interpreting natural language commands. It is designed to be embedded into interactive systems or agents that need to understand and act on textual instructions in human language.

## 🧠 What Is TorchMind?

TorchMind is a small feedforward neural network that classifies user input (in natural language) into predefined **intents**. Each intent corresponds to an action that your system can perform. The model is trained live in memory and can learn new commands dynamically without saving or reading from disk.

## 🚀 Key Features

- **Live Training**: The model learns new examples instantly using `.add_example()` and updates itself in memory.
- **Real-Time Prediction**: Converts user input to intent within milliseconds.
- **No File Dependency**: All logic is held in memory — no need for pre-saved JSONs.
- **Bag-of-Words Embedding**: Uses token-based BoW to represent input text.
- **Modular Design**: Easily integrated into any system (game agent, chatbot, UI layer, etc.) via a socket or direct API.

## ⚙️ How It Works

1. **Tokenization**: Input text is tokenized using NLTK.
2. **Bag of Words Vectorization**: Tokens are turned into a sparse vector.
3. **Classification**: A simple neural network predicts an intent label.
4. **Action Mapping**: Each label maps to an associated code block (JS string, API call, etc.).

Example:

```python
ai = CommandAI()
ai.add_example("go forward", "move_forward", "bot.setControlState('forward', true);")
ai.get_js("please move ahead")  
# ➜ "bot.setControlState('forward', true);"
```

## 📚 API

### `CommandAI.add_example(text, label, js_code)`
Adds a training example. Triggers retraining of the model.

### `CommandAI.train()`
Explicitly retrains the model from current examples (called automatically after `add_example`).

### `CommandAI.predict(text)`
Returns the predicted label for a given sentence.

### `CommandAI.get_js(text)`
Returns the action (JS code or string) associated with the predicted label.

## 🧪 Example Use Case

```python
ai = CommandAI()
ai.add_example("jump", "jump", "bot.setControlState('jump', true);")
print(ai.get_js("please jump now"))
# Output: "bot.setControlState('jump', true);"
```

## 🔍 Requirements

- Python 3.8+
- PyTorch
- NLTK (`nltk.download('punkt')` required)
- Optional: `socket` if used in a server/client architecture

## 🧩 Model Structure

- **Input**: Bag-of-Words vector (sparse)
- **Network**: 2 or 3 dense layers with ReLU activation
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam

## 🎯 Recommended Use Cases

- Game agents
- Smart assistants
- Custom command processors
- Embedded language interpreters

---

TorchMind is built for fast learning, natural control, and zero-delay understanding of human commands in text form. Use it anywhere you need quick, intelligent language-to-action mapping.
