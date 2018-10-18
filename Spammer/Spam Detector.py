import os
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn import metrics

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    # Paste code for non-word removal here(code snippet is given below)
    return dictionary

def extract_features(mail_dir,dictionary):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix

train_dir= r'C:\Python Projects\Spam Management\train-mails'
test_dir = r'C:\Python Projects\Spam Management\test-mails'
test_d=r'C:\Python Projects\Spam Management\x'
dict = make_Dictionary(train_dir)
train_matrix=extract_features(train_dir,dict)
# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1


# Training SVM and Naive bayes classifier

model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(train_matrix, train_labels)
model2.fit(train_matrix, train_labels)

# Test the unseen mails for Spam
test_matrix = extract_features(test_d,dict)
test_labels = np.zeros(1)
test_labels[0] = 1
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
print(result1)
print(result2)
print("Confusion matrix for MultinominalNB\n")
print(metrics.confusion_matrix(test_labels,result1))
print("Confusion matrix for SVM\n")
print(metrics.confusion_matrix(test_labels,result2))