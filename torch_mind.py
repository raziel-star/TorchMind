import torch
import torch.nn as nn

# === עיבוד טקסט – גרסה משופרת ===
def bag_of_words(tokenized_sentence, all_words):
    vector = torch.zeros(len(all_words), dtype=torch.float32)
    word_to_index = {word: idx for idx, word in enumerate(all_words)}
    for word in tokenized_sentence:
        index = word_to_index.get(word)
        if index is not None:
            vector[index] = 1.0
    return vector

# === TorchMind – רשת עצבית חיה עם בסיס ללמידה ===
class TorchMind(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# === פונקציית עזר: טוען מודל מאומן מקובץ .pth ===
def load_model(path: str):
    data = torch.load(path)
    model = TorchMind(data["input_size"], data["hidden_size"], data["output_size"])
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, data["all_words"], data["tags"]

# === פונקציית עזר: חיזוי תגית (intent) ממשפט ===
def predict_class(sentence, model, all_words, tags, threshold=0.75):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(sentence)
    bow = bag_of_words(tokens, all_words)
    output = model(bow)
    probs = torch.softmax(output, dim=0)
    prob, predicted = torch.max(probs, dim=0)

    if prob.item() >= threshold:
        return tags[predicted.item()]
    else:
        return "unknown"


# === הפעלה לבדיקה מקומית ===
if __name__ == "__main__":
    print("TorchMind loaded. This file is for model definition only.")
    try:
        model, all_words, tags = load_model("model.pth")
        print("✅ מודל נטען בהצלחה! בדיקת תחזית:")
        while True:
            sentence = input("👤 כתוב פקודה: ")
            if sentence == "יציאה":
                break
            tag = predict_class(sentence, model, all_words, tags)
            print(f"🤖 TorchMind חושב שזה שייך לקטגוריה: {tag}")
    except Exception as e:
        print("⚠️ לא ניתן לטעון מודל מאומן (model.pth). ודא שקיים קובץ כזה.")

   
