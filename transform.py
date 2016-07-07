from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from collections import Counter
#Parameters
datafn = "./data/training_data.txt"
labelfn = "./data/training_labels.txt"
testfn = "./data/test_data.txt"
numHashedFeatures = 2**14


def get_titles_and_labels(datafn, labelfn, runStats=True):
  """
  """
  print "Loading Titles and Labels To Be Used For Training Data..."
  with open(datafn, 'r') as f, open(labelfn, 'r') as g:
    numbers, titles = [], []
    labels = []
    for line, label_line in zip(f, g):
      info = line.split(',')
      numbers += info[0],
      trx_labels = [curLabel.strip() for curLabel in label_line.split(",")] 
      labels += trx_labels
      titles += [(" ".join(info[1:])).strip().split()] * len(trx_labels)
    if runStats:
      lCounter = Counter(labels)
      for label in sorted(lCounter, key=lCounter.get, reverse=True)[:20]:
        print label, lCounter[label]
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
  myMod = LogisticRegression(penalty="l1", C = 1.0 ) #[10**i for i in range(-2, 3)])
  myMod.fit(encoded_titles, encoded_labels)
  return myMod


def test_model(model, hasher, encoder, testfn=None):
  print "Testing the Model..."
  test_titles = []
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
  print labels
  return encoder.inverse_transform(labels)

titles, labels = get_titles_and_labels(datafn, labelfn)
t, l, h, e =   encode_titles_labels(titles, labels)
mod =  build_lr_model(t,l)
print test_model(mod, h, e, testfn)
