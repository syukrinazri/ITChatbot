#To generate random number
import random
#To use json file
import json
#Pickle trained data so that do not need to train each time run chatbot
import pickle
import numpy as np
#Package that is used to lemmatize and tokenize word (Preprocessing)
import nltk
from nltk.stem import WordNetLemmatizer

#Import tensorflow packages component that is needed to train algo
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

intents = json.loads(open('knowledgebase.json').read())

# Initiate list of words, classes and documents
words = []
classes = []
documents = []
# To get rid of certain symbols or word to make the algo differentiate the words better
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        
words = [WordNetLemmatizer().lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

#Write the words and classes to a pickled binary data
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#words cannot be processed directly using tensor flow, so, have to change the words into binary numbers that be processed by algorithm
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [WordNetLemmatizer().lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)        
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


random.shuffle(training)
training = np.array(training)

#Seperate data in order to train it with the algo
train_x = list(training[:, 0])
train_y = list(training[:, 1])


#Fit data to a Neural network layer model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

sgd = SGD(lr= 0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Save model into a h5 data to use for the chatbot
hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 5)
model.save('chatbotmodel.h5', hist)

print("Model done")


