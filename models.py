import pandas as pd
import csv

csv.field_size_limit(10**7)

sample_data = pd.read_csv("D:\Year 3\Text analytics and sentiment analysis\Env\sample_data_2.csv", engine='python')

sample_data = sample_data.dropna(subset=['label'])

# Combine the title and text into a new variable
label = sample_data['label']
combined_text =  sample_data['title'] + " " +  sample_data['text_new'].fillna("")

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(combined_text, label, test_size=0.2, random_state=42)

print('X train dataset', X_train.shape)
print('X test dataset', X_test.shape)
print('y train dataset', y_train.shape)
print('y test dataset', y_test.shape)

"""We uses the TF-IDF Vectorization to extract the features from the text data that can be used for text classification models training"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Fit and transform the training data to create TF-IDF vectors
X_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test = tfidf_vectorizer.transform(X_test)

"""# Logistic Regression
Logistic regression can be used for multi-class text classification tasks, assigning a given text to one of the several possible classes or categories. Since our dataset contains a binary output, it is totally applicable to use logistics regression to determine whether an input is a fake news or not.

To implement the predictive model, we will be first importing the required libraries. In this case, we will be building the model utilizing the Sklearn library. Since we have already applied the bag-of-words representation during data preparation, we can start training the model and test it.
"""

# Import required libraries
from sklearn.linear_model import LogisticRegression

# Train a logistic regression classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
lr_y_pred = lr.predict(X_test)

"""Lastly, let's evaluate the accuracy of the classifier."""

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Evaluate the model's performance
lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_report = classification_report(y_test, lr_y_pred, target_names=['Real', 'Fake'])
lr_cm = confusion_matrix(y_test, lr_y_pred)

# Specificity
tn, fp, fn, tp = lr_cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

# Calculate AUC-ROC
lr_y_prob = lr.predict_proba(X_test)[:, 1]

auc_roc = roc_auc_score(y_test, lr_y_prob)
print("AUC-ROC:", auc_roc)

# Display the model's performance
print("Accuracy:", lr_accuracy)
print(lr_report)

# Display confusion matrix
lr_disp = ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels=lr.classes_)
lr_disp.plot(cmap=plt.cm.Blues)
plt.xlabel('Predicted Class')
plt.ylabel('Real Class')
plt.title('Confusion Matrix for Logistics Regression')
plt.show()

"""As the result, it shows that the classifier has an accuracy score of 93.85%, approximately 94%.

# Random Forest
Random Forest is a powerful machine-learning model that can be used for supervised text classification tasks. It can effectively capture the key points by identifying which phrases are most useful in differentiating across classes (i.e. Fake News and Real News).

To implement the Random Forest Classification model, we will first import the required libraries by utilizing the Sklearn.
"""

# Import necessary library for Random Forest
from sklearn.ensemble import RandomForestClassifier

"""We can now train the Random Forest classifier model and start making predictions on the test set."""

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, random_state=42)

# Train the classifier
rf.fit(X_train, y_train)

# Predict the labels on the transformed test set
rf_y_pred = rf.predict(X_test)

"""Lastly, evaluate the accuracy of the Random Forest classifier."""

# Import necessary libraries for model evaluation and visualization
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Evaluate the model's performance
rf_accuracy = accuracy_score(y_test, rf_y_pred)                                         # Calculate accuracy of the predictions
rf_report = classification_report(y_test, rf_y_pred, target_names=['Real', 'Fake'])     # Generate a report which includes precision, recall, f1-score for 'Real' and 'Fake' classes
rf_cm = confusion_matrix(y_test, rf_y_pred)                                             # Generate the confusion matrix from the true and predicted labels

# Specificity
tn, fp, fn, tp = rf_cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

# Calculate AUC-ROC
rf_y_prob = rf.predict_proba(X_test)[:, 1]

auc_roc = roc_auc_score(y_test, rf_y_prob)
print("AUC-ROC:", auc_roc)


# Display the model's performance
print("Accuracy:", rf_accuracy)
print(rf_report)

# Display confusion matrix
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=rf.classes_)    # Initialize the confusion matrix display
rf_disp.plot(cmap=plt.cm.Blues)                                                         # Create a plot of the confusion matrix with a blue color map
plt.xlabel('Predicted Class')                                                           # X-axis of the visualization
plt.ylabel('Real Class')                                                                # Y-axis of the visualization
plt.title('Confusion Matrix for Random Forest')                                         # Title of the visualization
plt.show()

"""# Support Vector Machine (SVM)

To implement the SVM Classification model, we will first import the required libraries by utilizing the Sklearn.
"""

# Import necessary libraries for SVM
from sklearn.svm import LinearSVC, SVC

"""We can now train the SVM classifier model and start making predictions on the test set."""

# Initialize the SVM classifier
svm_classifier = SVC(probability=True)

# Train the classifier on the transformed training data
svm_classifier.fit(X_train, y_train)

# Predict the labels on the transformed test set
svm_y_pred = svm_classifier.predict(X_test)

"""Lastly, evaluate the accuracy of the SVM classifier."""

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Evaluate the model's performance
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_report = classification_report(y_test, svm_y_pred, target_names=['Real', 'Fake'])
svm_cm = confusion_matrix(y_test, svm_y_pred)

# Specificity
tn, fp, fn, tp = svm_cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

# Calculate AUC-ROC
svm_y_prob = svm_classifier.predict_proba(X_test)[:, 1]

auc_roc = roc_auc_score(y_test, svm_y_prob)
print("AUC-ROC:", auc_roc)


# Display the model's performance
print("Accuracy:", svm_accuracy)
print(svm_report)

# Display confusion matrix
svm_disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=svm_classifier.classes_)
svm_disp.plot(cmap=plt.cm.Blues)
plt.xlabel('Predicted Class')
plt.ylabel('Real Class')
plt.title('Confusion Matrix for SVM')
plt.show()

"""# Long Short Term Memory (LSTM)

To implement the LSTM model, we will first need to convert the text into One Hot Representation for the given vocabulaty size (i.e. 10000). But before that we will import all the necessary libraries, this time we will be mainly using tensorflow.
"""

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM,Bidirectional
from tensorflow.keras.layers import Dense, Dropout

# converting text into One Hot Representation
vocab_size = 10000
onehot_repr=[one_hot(words, vocab_size) for words in combined_text]

"""Next, we will perform embedding process. Paddings were added to the sentences so that all sentences are of same length i.e. 50, to avoid varying input sizes"""

sentence_length = 50

embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen=sentence_length)

"""At this point, we can start building the LSTM model. Firstly, we will have a vect or representation to text with only 100 features. Then, we can set up the model and connect the LSTM layer to the Dense layer."""

embedding_vector_features=100

model=Sequential()
model.add(Embedding(vocab_size, embedding_vector_features, input_length=sentence_length))
model.add(LSTM(100))
model.add(Dropout(0.2))

# LSTM layer (output) is fully connected to the Dense layer
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

import numpy as np
X_final = np.array(embedded_docs)
y_final = np.array(label)
X_final.shape, y_final.shape

# Splitting the data for training and validating
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Testing the model
y_log = model.predict(X_test)
lstm_y_pred = np.where(y_log>0.5,1,0)

# Calculate accuracy and generating classification report and confusion matrix
lstm_accurary = accuracy_score(y_test, lstm_y_pred)

lstm_report = classification_report(y_test, lstm_y_pred, target_names=['Real', 'Fake'])

lstm_cm = confusion_matrix(y_test, lstm_y_pred)

# Specificity
tn, fp, fn, tp = lstm_cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

# Calculate AUC-ROC
# lstm_y_prob = model.predict_proba(X_test)[:, 1]

auc_roc = roc_auc_score(y_test, y_log)
print("AUC-ROC:", auc_roc)

# Display accuracy and classification report
print('Accuracy: ', lstm_accurary)
print(lstm_report)

# Display confusion matrix
lstm_disp = ConfusionMatrixDisplay(confusion_matrix=lstm_cm)
lstm_disp.plot(cmap=plt.cm.Blues)
plt.xlabel('Predicted Class')
plt.ylabel('Real Class')
plt.title('Confusion Matrix for LSTM')
plt.show()