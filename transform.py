"""
Read in some training data and labels
Transform titles into numerical values
Encode labels into numerical values (only used one routing label for now)
Train a classification model
Perform validation to approximate real-world performance
Test of new data
"""
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from collections import Counter


from label_stats import IssueStats
#Parameters
datafn = "./data/training_data.txt"
labelfn = "./data/training_labels.txt"
testfn = "./data/test_data.txt"
numHashedFeatures = 2**16


def get_titles_and_labels(datafn, labelfn, maxCount=5000):
  """
  """
  print "Loading Titles and Labels To Be Used For Training Data..."
  titles, labels = [], [] 
  iStats = IssueStats()
  for i, data in enumerate(iStats.get_training_examples()):
    titles += data[0].split(),
    labels += data[1],
  return titles, labels


def encode_titles_labels(titles, labels):
  print "Encoding titles and labels to numerical values..."
  myHasher = FeatureHasher(input_type="string", n_features= numHashedFeatures, non_negative=True)
  myEncoder = LabelEncoder()
  featureMatrix = myHasher.transform(titles)
  myEncoder.fit(labels)
  encoded = myEncoder.transform(labels)
  return featureMatrix, encoded, myHasher, myEncoder

def build_lr_model(encoded_titles, encoded_labels):
  print "Training a Model..."
  #by default 3 fold cv
  #myMod = LogisticRegression(C=100., penalty='l1') 
  myMod = SGDClassifier(loss='log', penalty='l2', alpha = .001) #, alpha = .0001, class_weight = 'balanced') 
  #myMod = LassoCV(alphas = [10**i for i in range(-3,4)])  #[10**i for i in range(-2, 3)]) 
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
  print model.score(testMat, testLabels)
  return encoder.inverse_transform(labels)

def count_words(titles):
  mySet = set()
  for title in titles:
    for word in title:
      word = word.translate(None, ">(*&^%$#@!{},.")
      mySet.add(word)
  print len(mySet), "unique words"

if __name__ == "__main__":
  titles, labels = get_titles_and_labels(datafn, labelfn)
  t, l, h, e =   encode_titles_labels(titles, labels)
  train_mat, test_mat, train_labels, test_labels = train_test_split(t, l, random_state=32)
  mod = build_lr_model(train_mat, train_labels)
  test_model(mod, h, e, testMat=test_mat, testLabels=test_labels)
