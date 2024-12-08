import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Lists to hold the data
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Load intents data from the intents.json file
with open('intents.json') as file:
    intents = json.load(file)

# Process the intents and store them
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add document (tokenized pattern) to the documents list
        documents.append((w, intent['tag']))
        # Add the intent tag to the classes list if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lowercase words, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # Remove duplicates and sort the words

# Sort classes
classes = sorted(list(set(classes)))

# Print summary of processed data
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Save the words and classes using pickle
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create the training data
training = []
output_empty = [0] * len(classes)  # Empty output array for one-hot encoding

# Create a bag of words for each sentence
for doc in documents:
    # Initialize the bag of words
    bag = []
    pattern_words = doc[0]
    # Lemmatize each word - create base word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create the bag of words for the current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Check if the bag length matches the number of words
    if len(bag) != len(words):
        print(f"Warning: Bag length mismatch. Expected: {len(words)}, Got: {len(bag)}")

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Check the consistency of the output row length
    if len(output_row) != len(classes):
        print(f"Warning: Output row length mismatch. Expected: {len(classes)}, Got: {len(output_row)}")

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)
training = np.array(training, dtype=object)  # Explicitly set dtype to object for inconsistent lengths

# Split the training data into input (X) and output (Y)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print("Training data created")

# Create the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('model.h5')
print("Model created and saved as 'model.h5'")
