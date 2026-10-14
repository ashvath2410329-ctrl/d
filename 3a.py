from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, classification_report

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200, padding="post").astype("float32") / 10000
x_test = pad_sequences(x_test, maxlen=200, padding="post").astype("float32") / 10000

inp = Input(shape=(200,))
h = Dense(128, activation="relu")(inp)
h = Dense(64, activation="relu")(h)
latent = Dense(32, activation="relu")(h)
h = Dense(64, activation="relu")(latent)
h = Dense(128, activation="relu")(h)
out = Dense(200, activation="sigmoid")(h)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(x_train, x_train, epochs=5, batch_size=128, validation_split=0.2)

encoder = Model(inp, latent)
f_train = encoder.predict(x_train, verbose=0)
f_test = encoder.predict(x_test, verbose=0)

classifier = Sequential([
    Dense(16, activation="relu", input_shape=(32,)),
    Dense(1, activation="sigmoid"),
])
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit(f_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

y_prob = classifier.predict(f_test, verbose=0)
y_pred = (y_prob > 0.5).astype(int).ravel()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
