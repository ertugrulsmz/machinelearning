import json 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("dataset/sarcasm.json","r") as jsonfile:
    datastore = json.load(jsonfile)
    
sentences = []
labels = []

for jsoniterator in datastore:
    sentences.append(jsoniterator['headline']) #sentences are here
    labels.append(jsoniterator['is_sarcastic']) #binary value    
    

#hyperparameters
vocab_size = 1000
embedding_dim = 16
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 18000 

#adjust training set
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


#Encode sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)  

#learning process
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 

model.summary()

epoch_quantity = 20
history = model.fit(training_padded, training_labels, epochs=epoch_quantity, 
                    validation_data=(testing_padded, testing_labels))    

#history keeps the accuracy and loss values step by step as a array.

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


#reviewing words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text]) #? occurs voc not enough

print(decode_sentence(training_padded[0]))
print(training_sentences[2])
print(labels[2])



e = model.layers[0]
weights = e.get_weights()[0]
#like table where each word has weights by the size of embedding_dim
# if sequence 1,5,55, ? then we have 3 vector each has embedding_dim weights in table
#those values will be like input thought.

print(weights.shape) # shape: (vocab_size, embedding_dim)


#input test
inputsentences = ["feel like a bit overrated", 
            "you should watch before you die"]

sequences = tokenizer.texts_to_sequences(inputsentences)

padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(model.predict(padded))







    