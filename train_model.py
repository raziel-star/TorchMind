import json
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
from torch_mind import TorchMind, bag_of_words

nltk.download('punkt')

# === שלב 1: קריאה של intents.json ===
file_path = "intents.json"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        print("=== הקובץ נקרא בהצלחה ===")
        print(content[:200] + "..." if len(content) > 200 else content)

        intents = json.loads(content)
        print("✅ JSON תקין!")
        print("מצאתי", len(intents["intents"]), "intents.")
except Exception as e:
    print("❌ שגיאה בטעינת הקובץ:", e)
    exit()

# === שלב 2: עיבוד מילים + תגיות ===
all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        words = word_tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))

ignore_words = ["?", "!", ".", ","]
all_words = [w for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# === שלב 3: הפיכת טקסט לוקטורים ===
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bow = bag_of_words(pattern_sentence, all_words)
    X_train.append(bow)
    label = tags.index(tag)
    y_train.append(label)

X_train = torch.stack(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

# === שלב 4: בניית המודל ואימון ===
input_size = len(all_words)
hidden_size = 16
output_size = len(tags)
model = TorchMind(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 1000

for epoch in range(epochs):
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"🧠 Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

print("✅ האימון הסתיים!")

# === שלב 5: שמירת המודל המאומן ===
torch.save({
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}, "model.pth")

print("💾 המודל נשמר בהצלחה בשם: model.pth")
