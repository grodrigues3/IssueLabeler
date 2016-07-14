"""
1. Read in some training data and labels
2. Transform titles into numerical values
3. Encode labels into numerical values (only used one routing label for now)
4. Train a classification model
5. Perform validation to approximate real-world performance
6. Test of new data
"""
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time
from collections import Counter
from label_stats import IssueStats


import pdb
#Parameters
datafn = "./data/training_data.txt"
labelfn = "./data/training_labels.txt"
testfn = "./data/test_data.txt"
with open("./data/stopwords.txt", 'r') as f:
  stopwords = [word.strip() for word in f]


def get_titles_and_labels(datafn, labelfn, maxCount=5000):
  """
  Load in the body and title
    - remove punctuation
    - remove stopwords
    - stem the existing words
  Load in the labels

  """
  print "Loading Titles and Labels To Be Used For Training Data..."
  titles_and_body, labels = [], [] 
  iStats = IssueStats()
  myStemmer = PorterStemmer()
  tokenizer = RegexpTokenizer(r'\w+')
  for i, data in enumerate(iStats.get_training_examples()):
    title = tokenizer.tokenize(data[0].decode('utf-8').lower())  
    titles_and_body += map(myStemmer.stem, filter(lambda x: x not in stopwords, title)), # title.split())),
    labels += data[1],
  return titles_and_body, labels


def encode_titles_labels(titles, labels, numFeatures= 2**14):
  print "Encoding titles and labels to numerical values..."
  myHasher = FeatureHasher(input_type="string", n_features= numFeatures, non_negative=True)
  myEncoder = LabelEncoder()
  featureMatrix = myHasher.transform(titles)
  myEncoder.fit(labels)
  encoded = myEncoder.transform(labels)
  return featureMatrix, encoded, myHasher, myEncoder

def build_lr_model(encoded_titles, encoded_labels, myLoss= 'log', myPenalty = 'l2', myAlpha = .0001):
  print "Training a Model..."
  myMod = SGDClassifier(loss=myLoss, penalty=myPenalty, alpha = myAlpha) #, alpha = .0001, class_weight = 'balanced') 
  myMod.fit(encoded_titles, encoded_labels)
  return myMod

def test_model(model, hasher, encoder, testfn=None, testMat=None, testLabels=None):
  print "Testing the Model..."
  test_titles = []
  if testMat is None:
    if testfn is not None: 
      with open(testfn, 'r') as f:
        for line in f:
          info = line.split(",")
          test_titles += [(" ".join(info[1:])).strip().split()]
      test_titles = test_titles[:10]
    else:
      cont = 1
      while cont:
        test_in = raw_input("Give me a sample issue title")
        test_titles += test_in.split(),
        cont = [0,1][raw_input("Add another issue title to test?")[0] == 'y']
    testMat = hasher.transform(test_titles)
  labels = model.predict(testMat)
  #encoder.inverse_transform(labels)
  return model.score(testMat, testLabels)

def count_words(titles):
  mySet = set()
  for title in titles:
    for word in title:
      mySet.add(word)
  print len(mySet), "unique words"


def save_sparse(fn, titleMatrix, labels, hasher, encoder):
  """
  Store a local copy of the sparse matrix on disk so that we don't
  have to perform punctuation stripping, stopword removal, and feature hashign
  repeatedly
  """
  data_dict = {"indices": titleMatrix.indices,
               "indptr" : titleMatrix.indptr,
               "shape"  : titleMatrix.shape,
               "data"   : titleMatrix.data,
               "encoder" : encoder,
               "hasher" : hasher,
               "labels" : labels}
  np.savez(fn, **data_dict)


def load_sparse_csr(fn):
  myLoader = np.load(fn)
  return csr_matrix((  myLoader['data'], myLoader['indices'], myLoader['indptr']),
                    shape = myLoader['shape']), myLoader["labels"], myLoader["hasher"], myLoader["encoder"].item()



def perform_cross_validation():
  best = 0.0
  best_params = None
  c = 0
  toDo = len(numFeatures) * len(alphas) * len(penalties) * len(losses)
  numFeatures = [2**i for i in range(15, 20)]
  alphas = [10**i for i in range(3, -4, -1)]
  penalties = ['l1', 'l2', 'elasticnet']
  losses = ['log', 'hinge']
  #bookkeeping prep
  #g = open('cv_finer_tuned.csv', 'w')
  #g.write("NumFeatures, Loss, Alpha, Penalty, Score\n")
  for nn in numFeatures:
    t, l, h, e =  encode_titles_labels(titles, labels, numFeatures = nn)
    train_mat, test_mat, train_labels, test_labels = train_test_split(t, l, random_state=int(time.time()), test_size=.2)
    for loss in losses:
      for penalty in penalties:
        for alpha in alphas:
          #build and train the model
          mod = build_lr_model(train_mat, train_labels, myLoss=loss, myPenalty=penalty, myAlpha=alpha)
          score = test_model(mod, h, e, testMat=test_mat, testLabels=test_labels)

          #bookkeeping
          params = (nn, loss, alpha, penalty, score)
          if score > best:
            best = score
            best_params = params
          #g.write(("{},"*5 +"\n").format(*params) ) 
          c += 1
          if c % 5 == 0:
            print c, 'models trained out of', toDo
  #g.close()
  print best_params
  print best



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(iris.target_names))
      plt.xticks(tick_marks, iris.target_names, rotation=45)
      plt.yticks(tick_marks, iris.target_names)
      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')

if __name__ == "__main__":
  #titles, labels = get_titles_and_labels(datafn, labelfn)
  #count_words(titles)
  #t, l, h, e =  encode_titles_labels(titles, labels, numFeatures = 8192)
  data_persistence = "tMat_8192.npz"
  #save_sparse(data_persistence, t, l, h, e)
  tMat, labels, h, e = load_sparse_csr(data_persistence)
  train_mat, test_mat, train_labels, test_labels = train_test_split(tMat, labels, random_state=int(time.time()), test_size=.2)
  best_params = 'hinge', 'l2', .01, 
  mod = build_lr_model(train_mat, train_labels, *best_params) #myLoss=loss, myPenalty=penalty, myAlpha=alpha)
  predicted_labels = mod.predict(test_mat)
  true_txt_labels = e.inverse_transform(test_labels)
  predicted_txt_labels = e.inverse_transform(predicted_labels)
  for x,y in zip(true_txt_labels, predicted_txt_labels):
    print x,y
  cm = confusion_matrix(true_txt_labels, predicted_txt_labels)
  pdb.set_trace()
  print cm
