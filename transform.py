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
from scipy.sparse import csr_matrix
import time
from collections import Counter
from get_issues_labels import IssueStats

# Parameters
datafn = "./data/training_data.txt"
labelfn = "./data/training_labels.txt"
testfn = "./data/test_data.txt"
DATAPERSISTENCE = "persistence/tMat_compOnly.npz"
PICKLEFN = "pkld/trained_components_model.pkl" #"pkld/trained_teams_model.pkl"
myStemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stopwords = None

try:
  if not stopwords:
    stop_fn = "./data/stopwords.txt"
  with open(stop_fn, 'r') as f:
    stopwords = [word.strip() for word in f]
except:
  #don't remove any stopwords
  stopwords = []

def get_labels():
  print "Loading Just The Labels..."
  labels = []
  iStats = IssueStats()
  counts = defaultdict(lambda:0)
  for i, data in enumerate(iStats.get_training_examples(which_label='component')):
    labels += data[1],
    counts[labels[-1]] += 1
  return labels, counts

def tokenize_stem_stop(inputString):
    curTitleBody = tokenizer.tokenize(inputString.decode('utf-8').lower())
    return map(myStemmer.stem, filter(lambda x: x not in stopwords, curTitleBody))

def get_titles_and_labels(which_label='team'):
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
  for i, data in enumerate(iStats.get_training_examples(which_label=which_label)):
    titles_and_body += tokenize_stem_stop(data[0]),
    labels += data[1],
  return titles_and_body, labels


def encode_titles(titles, numFeatures= 2**14):
  """
  Encode the titles formatted as a string as numerical values using
  the "hashing trick".  The size of the feature vector can be specified using the
  numFeatures parameter"
  """
  print "Encoding titles and labels to numerical values..."
  myHasher = FeatureHasher(input_type="string", n_features= numFeatures, non_negative=True)
  featureMatrix = myHasher.transform(titles)
  return featureMatrix, myHasher#, myEncoder


def build_lr_model(encoded_titles, encoded_labels, myLoss= 'log', myPenalty = 'l2', myAlpha = .0001):
  #print "Training a Model..."
  myMod = SGDClassifier(loss=myLoss, penalty=myPenalty, alpha = myAlpha) 
  myMod.fit(encoded_titles, encoded_labels)
  return myMod

def test_model(model, hasher, testfn=None, testMat=None, testLabels=None):
  #print "Testing the Model..."
  """
  Three ways to test the existing model
    1) pass in a testMat: a with the same column dimensionality as the training matrix
    2) pass in a testFn
    3) type your own title and body in to the raw_input field
  """
  test_titles = []
  if testMat is None:
    if testfn is not None: 
      with open(testfn, 'r') as f:
        for line in f:
          info = line.split(",")
          test_titles += tokenize_stem_stop(info[1]), 
    else:
      cont = 1
      while cont:
        test_in = raw_input("Give me a sample issue title")
        test_titles += tokenize_stem_stop(test_in),
        cont = [0,1][raw_input("Add another issue title to test?")[0] == 'y']
    testMat = hasher.transform(test_titles)
  return model.score(testMat, testLabels)

def count_words(titles):
  """
  To estimate the size of the feature vector as passed into the hashing trick,
  count the vocabulary size (the number of unique tokens used in all titles and bodies)
  """
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
  numFeatures = [2**i for i in range(15, 20)]
  alphas = [10**i for i in range(3, -5, -1)]
  penalties = ['l2', 'elasticnet']
  losses = ['log', 'hinge']
  toDo = len(numFeatures) * len(alphas) * len(penalties) * len(losses)
  #bookkeeping prep
  g = open('cv_finer_tuned_teams.csv', 'w')
  g.write("NumFeatures, Loss, Alpha, Penalty, Score\n")
  for nn in numFeatures:
    trainingMat, myHasher =  encode_titles(titles, numFeatures = nn)
    train_mat, test_mat, train_labels, test_labels = train_test_split(trainingMat, labels, random_state=int(time.time()), test_size=.2)
    for penalty in penalties:
      for loss in losses:
        for alpha in alphas:
          #build and train the model
          mod = build_lr_model(train_mat, train_labels, myLoss=loss, myPenalty=penalty, myAlpha=alpha)
          score = mod.score(test_mat, test_labels) #test_model(mod, myHasher, testMat=test_mat, testLabels=test_labels)
          #bookkeeping
          params = [nn, loss, alpha, penalty]
          if score > best:
            best = score
            best_params = params
          toWrite = params +[score]
          g.write(("{},"*len(params) +"{}\n").format(*toWrite) )
          c += 1
          if c % 10 == 0:
            print "Loss, Penalty, Alpha:{:<10}\t{:10}\t{:<10}\t{:2.3f}".format(loss, penalty, alpha, score)
            print c, 'models trained out of', toDo
  return best_params, best

def save_model(mod, PICKLEFN):
  return joblib.dump(mod,PICKLEFN)

def load_model(PICKLEFN):
  return joblib.load(PICKLEFN)

if __name__ == "__main__":
  titles, labels = get_titles_and_labels(which_label='component')
  print len(titles), titles[:10]
  print len(labels), labels[:10]
  """
  #save_sparse(DATAPERSISTENCE, tMat, labels, hasher)
  #best_params, best_score = perform_cross_validation(titles, labels)
  #print best_params, best_score
  mod = load_model(PICKLEFN)
  best_params = (262144, 'hinge', 0.1, 'l2') #for team and copmonents
  #best_comp_params =[262144, 'hinge', 0.1, 'l2'] 
  nFeats, bLoss, bAlpha, bPen = best_params
  tMat, hasher = encode_titles(titles, nFeats)
  save_sparse(DATAPERSISTENCE, tMat, labels, hasher)
  #tMat, labels, hasher = load_sparse_csr(DATAPERSISTENCE)
  train_mat, test_mat, train_labels, test_labels = train_test_split(tMat, labels, random_state=int(time.time()), test_size=.2)
  mod = build_lr_model(train_mat, train_labels, myLoss=bLoss, myPenalty=bPen, myAlpha=bAlpha) 
  #print mod.score(test_mat, test_labels)
  #save_model(mod, PICKLEFN) 
  predicted_labels = mod.predict(test_mat)
  print "Model Validation Score: ", mod.score(test_mat, test_labels)
  txt_labels = list(set(labels))
  cm = confusion_matrix(test_labels, predicted_labels, labels=txt_labels)
  labels, counts = get_labels()
  predicted_out = Counter(predicted_labels)

  print "Total Labels", sum(counts.values()), len(labels)
  total_labels = len(labels)
  print ("{:<30}\t"*3).format('Label', 'Real Dist', 'Predicted Dist')
  for thing in sorted(counts, key= counts.get, reverse=True):
    print ("{:<30}\t"*3).format(thing, counts[thing]*1./total_labels, predicted_out[thing]*1.0/sum(predicted_out.values()))
  _, _, myHasher = load_sparse_csr(DATAPERSISTENCE)
  myHasher = myHasher.item()
  if False:
    while 1:
      data = tokenize_stem_stop(raw_input("Give me an example issue body\n"))
      example = myHasher.transform([data])
      print mod.predict(example)
  """
