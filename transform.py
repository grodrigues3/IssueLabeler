"""
1. Read in some training data and labels
2. Transform titles into numerical values
3. Encode labels into numerical values (only used one routing label for now)
4. Train a classification model
5. Perform validation to approximate real-world performance
6. Test of new data
"""
from collections import defaultdict
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time
from collections import Counter
from get_issues_labels import IssueStats
import pdb



#Parameters
datafn = "./data/training_data.txt"
labelfn = "./data/training_labels.txt"
testfn = "./data/test_data.txt"
DATAPERSISTENCE = "persistence/tMat_8192.npz"
PICKLEFN = "pkld/trained_model.pkl"
with open("./data/stopwords.txt", 'r') as f:
  stopwords = [word.strip() for word in f]


def get_labels():
  """
  Load in the body and title
    - remove punctuation
    - remove stopwords
    - stem the existing words
  Load in the labels

  """
  print "Loading Just The Labels..."
  labels = []
  iStats = IssueStats()
  counts = defaultdict(lambda:0)
  for i, data in enumerate(iStats.get_training_examples()):
    labels += data[1],
    counts[labels[-1]] += 1
  return labels, counts

def get_titles_and_labels():
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


def encode_titles(titles, numFeatures= 2**14):
  print "Encoding titles and labels to numerical values..."
  myHasher = FeatureHasher(input_type="string", n_features= numFeatures, non_negative=True)
  #myEncoder = LabelEncoder()
  featureMatrix = myHasher.transform(titles)
  #myEncoder.fit(labels)
  #encoded = myEncoder.transform(labels)
  #don't have to encode labels for newer sklearn models, string labels are fine
  return featureMatrix, myHasher#, myEncoder


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


def save_sparse(fn, titleMatrix, labels, hasher):
  """
  Store a local copy of the sparse matrix on disk so that we don't
  have to perform punctuation stripping, stopword removal, and feature hashign
  repeatedly
  """
  data_dict = {"indices": titleMatrix.indices,
               "indptr" : titleMatrix.indptr,
               "shape"  : titleMatrix.shape,
               "data"   : titleMatrix.data,
               "hasher" : hasher,
               "labels" : labels}
  np.savez(fn, **data_dict)


def load_sparse_csr(fn):
  myLoader = np.load(fn)
  return csr_matrix(( myLoader['data'], myLoader['indices'], myLoader['indptr']),
                    shape = myLoader['shape']), myLoader["labels"], myLoader["hasher"]


def perform_cross_validation(titles, labels):
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
    trainingMat, myHasher =  encode_titles(titles, numFeatures = nn)
    train_mat, test_mat, train_labels, test_labels = train_test_split(trainingMat, labels, random_state=int(time.time()), test_size=.2)
    for loss in losses:
      for penalty in penalties:
        for alpha in alphas:
          #build and train the model
          mod = build_lr_model(train_mat, train_labels, myLoss=loss, myPenalty=penalty, myAlpha=alpha)
          score = test_model(mod, myHasher, e, testMat=test_mat, testLabels=test_labels)

          #bookkeeping
          params = (nn, loss, alpha, penalty, score)
          if score > best:
            best = score
            best_params = params
          #g.write(("{},"*5 +"\n").format(*params) ) 
          c += 1
          if c % 5 == 0:
            print c, 'models trained out of', toDo
    return best_params, best_score


def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(labels))
      print labels
      plt.xticks(tick_marks, labels, rotation=45)
      plt.yticks(tick_marks, labels)
      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      plt.show()


def save_model(mod, PICKLEFN):
  return joblib.dump(mod,PICKLEFN)

def load_model(PICKLEFN):
  return joblib.load(PICKLEFN)

if __name__ == "__main__":
  """
  titles, labels = get_titles_and_labels()
  trainingMat8192, hasher=  encode_titles(titles, numFeatures = 8192)
  save_sparse(DATAPERSISTENCE, trainingMat8192, labels, hasher)
  tMat, numericLabels, hasher = load_sparse_csr(DATAPERSISTENCE)
  train_mat, test_mat, train_labels, test_labels = train_test_split(tMat, labels, random_state=int(time.time()), test_size=.2)
  best_params = 'hinge', 'l2', .1, 
  mod = build_lr_model(train_mat, train_labels, *best_params) #myLoss=loss, myPenalty=penalty, myAlpha=alpha)
  predicted_labels = mod.predict(test_mat)
  txt_labels = list(set(get_labels()))
  cm = confusion_matrix(test_labels, predicted_labels, labels=txt_labels[:10])[:10, :10]
  plot_confusion_matrix(cm, txt_labels[:10])
  """
  labels, counts = get_labels()
  print "Total Labels", sum(counts.values()), len(labels)
  for thing in sorted(counts, key= counts.get, reverse=True):
    print thing, counts[thing]
  exit()

  _, _, myHasher = load_sparse_csr(DATAPERSISTENCE)
  myHasher = myHasher.item()
  pdb.set_trace()
  while 1:
    data = raw_input("Give me an example issue body\n")
    example = myHasher.transform([data])
    print mod.predict(example)
