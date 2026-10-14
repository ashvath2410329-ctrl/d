import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Embedding, LSTM, Add, Dropout

random.seed(42)
np.random.seed(42)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

class_names = ["t-shirt", "trouser", "pullover", "dress", "coat",
               "sandal", "shirt", "sneaker", "bag", "ankle boot"]
templates = ["a photo of a", "this picture shows a", "an image containing a",
             "a small picture of a", "here is a"]


def caption(label):
    return f"startseq {random.choice(templates)} {class_names[label]} endseq"


n_train, n_test = 4000, 500
train_idx = np.random.choice(len(x_train), n_train, replace=False)
test_idx = np.random.choice(len(x_test), n_test, replace=False)
train_images = x_train[train_idx]
train_captions = [caption(l) for l in y_train[train_idx]]
test_images = x_test[test_idx]
print("Example captions:", train_captions[:3])

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)
vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(s) for s in train_seqs)
print(f"Vocab size: {vocab_size}, Max caption length: {max_len}")

imgs, words, nxt = [], [], []
for img, seq in zip(train_images, train_seqs):
    for t in range(1, len(seq)):
        imgs.append(img)
        words.append(seq[:t])
        nxt.append(seq[t])

imgs = np.array(imgs)
words = pad_sequences(words, maxlen=max_len)
nxt = to_categorical(nxt, num_classes=vocab_size)
print("Training pairs:", imgs.shape[0])

image_in = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(image_in)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D()(x)
x = GlobalAveragePooling2D()(x)
eye = Dense(128, activation="relu")(x)
eye = Dropout(0.3)(eye)

words_in = Input(shape=(max_len,))
y = Embedding(vocab_size, 128, mask_zero=True)(words_in)
y = Dropout(0.3)(y)
reader = LSTM(128)(y)

z = Add()([eye, reader])
z = Dense(128, activation="relu")(z)
out = Dense(vocab_size, activation="softmax")(z)

caption_model = Model([image_in, words_in], out)
caption_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
caption_model.summary()

history = caption_model.fit([imgs, words], nxt, epochs=10, batch_size=64, validation_split=0.1)

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Caption model training loss")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross-entropy")
plt.legend()
plt.tight_layout()
plt.savefig("caption_training_loss.png")
plt.show()


def generate_caption(image):
    text = "startseq"
    for _ in range(max_len):
        ids = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_len)
        idx = np.argmax(caption_model.predict([image[None], ids], verbose=0))
        word = tokenizer.index_word.get(idx, "endseq")
        if word == "endseq":
            break
        text += " " + word
    return text.replace("startseq", "").strip()


fig, axes = plt.subplots(2, 3, figsize=(12, 8))
sample_indices = np.random.choice(len(test_images), 6, replace=False)
for ax, idx in zip(axes.ravel(), sample_indices):
    img = test_images[idx]
    true_label = class_names[y_test[test_idx[idx]]]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"True: {true_label}\nGenerated: {generate_caption(img)}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("caption_predictions.png")
plt.show()

print("Example generated caption:", generate_caption(test_images[0]))
