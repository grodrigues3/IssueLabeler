'''
1. Read in some training data and labels
2. Transform titles into numerical values
3. Encode labels into numerical values (only used one routing label for now)
4. Train a classification model
5. Perform validation to approximate real-world performance
6. Test of new data
'''
from collections import defaultdict, Counter
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
from scipy.sparse import csr_matrix
import time
from get_issues_labels import IssueStats

################################################################################
# Parameters
datafn = './data/training_data.txt'
labelfn = './data/training_labels.txt'
testfn = './data/test_data.txt'
DATAPERSISTENCE = 'persistence/training_mat_compOnly.npz'
PICKLEFN = 'pkld/trained_components_model.pkl'
myStemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stopwords = None

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
             'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
             'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
             'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
             'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
             'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to',
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
             'again', 'further', 'then', 'once', 'here', 'there', 'when',
             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
             'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
             'just', 'don', 'should', 'now']

################################################################################
def tokenize_stem_stop(inputString):
  '''
  Strip punctuation
  Remove stopwords
  Tokenize
  inputString: string
  return: list of strings (tokens)
  '''
  curTitleBody = tokenizer.tokenize(inputString.decode('utf-8').lower())
  return map(myStemmer.stem,
             filter(lambda x: x not in stopwords, curTitleBody))


def encode_titles(titles, num_features=2**14):
  '''
  Encode the titles formatted as a string as numerical values using
  the 'hashing trick'.
  The size of the feature vector can be specified using the
  num_features parameter'
  '''
  myHasher = FeatureHasher(input_type='string',
                           n_features= num_features,
                           non_negative=True)
  featureMatrix = myHasher.transform(titles)
  return featureMatrix, myHasher

def build_lr_model(encoded_titles, encoded_labels,
                   my_loss='log', my_penalty='l2', my_alpha=.0001):
  my_mod = SGDClassifier(loss=my_loss, penalty=my_penalty, alpha = my_alpha)
  my_mod.fit(encoded_titles, encoded_labels)
  return my_mod


def test_model(model, hasher, testfn=None,
               testraining_mat=None, test_labels=None):
  '''
  Three ways to test the existing model
    1) pass in a testraining_mat: a with the same column dimensionality as the training matrix
    2) pass in a testFn
    3) type your own title and body in to the raw_input field
  '''
  test_titles = []
  if testraining_mat is None:
    if testfn is not None:
      with open(testfn, 'r') as f:
        for line in f:
          info = line.split(',')
          test_titles += tokenize_stem_stop(info[1]),
    else:
      cont = True
      while cont:
        test_in = raw_input('Give me a sample issue title')
        test_titles += tokenize_stem_stop(test_in),
        cont = [0,1][raw_input('Add another issue title to test?')[0] == 'y']
    testraining_mat = hasher.transform(test_titles)
  return model.score(testraining_mat, test_labels)


def save_sparse(fn, titleMatrix, labels, hasher):
  '''
  Store a local copy of the sparse matrix on disk so that we don't
  have to perform punctuation stripping, stopword removal, and feature hashign
  repeatedly
  '''

  data_dict = {'indices': titleMatrix.indices,
               'indptr': titleMatrix.indptr,
               'shape': titleMatrix.shape,
               'data': titleMatrix.data,
               'hasher': hasher,
               'labels': labels}
  np.savez(fn, **data_dict)

def load_sparse_csr(fn):
  myLoader = np.load(fn)
  retMat = csr_matrix((myLoader['data'],
                       myLoader['indices'],
                       myLoader['indptr']),
                      shape=myLoader['shape']), 
  return retMat, myLoader['labels'], myLoader['hasher']


def perform_cross_validation(my_titles, my_labels):
  """
  """
  best_score = 0.0
  best_params = None
  c = 0
  num_features = [2**i for i in range(15, 20)]
  alphas = [10**i for i in range(3, -5, -1)]
  penalties = ['l2', 'elasticnet']
  losses = ['log', 'hinge']
  to_do = len(num_features) * len(alphas) * len(penalties) * len(losses)
  g = open('cv_finer_tuned_teams.csv', 'w')
  g.write('NumFeatures, Loss, Alpha, Penalty, Score\n')
  print 'Total:{} {:<10}\t{:10}\t{:<10}\t{}'.format(to_do, 'Loss', 'Penalty', 'Alpha', 'Score')
  for nn in num_features:
    training_matrix, my_hasher = encode_titles(my_titles, num_features=nn)
    training_matrix, test_matrix, training_labels, test_labels =
                          train_test_split(training_matrix,
                                           my_labels,
                                           random_state=int(time.time()),
                                           test_size=.2)
    for penalty in penalties:
      for loss in losses:
        for alpha in alphas:
          mod = build_lr_model(training_matrix,
                               training_labels,
                               my_loss=loss,
                               my_penalty=penalty,
                               my_alpha=alpha)
          score = mod.score(test_matrix, test_labels)
          params = [nn, loss, alpha, penalty]
          if score > best_score:
            best_score = score
            best_params = params
          to_write = params +[score]
          g.write(('{},'*len(params) +'{}\n').format(*to_write))
          c += 1
          if c % 10 == 0:
            print '{}:{:<10}\t{:10}\t{:<10}\t{:2.3f}'.format(c,
                                                             loss,
                                                             penalty,
                                                             alpha,
                                                             score)
  print best_params, best_score
  return best_params, best_score

if __name__ == '__main__':
  iStats = IssueStats()
  titles, labels = zip(*iStats.get_training_examples('team'))
  titles = [tokenize_stem_stop(title) for title in titles]
  best_params = (262144, 'hinge', 0.1, 'l2')
  num_feats, best_loss, best_alpha, best_pen = best_params
  # save_sparse(DATAPERSISTENCE, training_mat, labels, hasher)
  best_params, best_score = perform_cross_validation(titles, labels)

  # mod = load_model(PICKLEFN) #best_comp_params =[262144, 'hinge', 0.1, 'l2']
  training_mat, hasher = encode_titles(titles, num_feats)
  # save_sparse(DATAPERSISTENCE, training_mat, labels, hasher)
  # training_mat, labels, hasher = load_sparse_csr(DATAPERSISTENCE)
  split_data = train_test_split(training_mat,
                                labels,
                                random_state=int(time.time()),
                                test_size=.2)
  train_mat, test_mat, train_labels, test_labels = split_data
  mod = build_lr_model(train_mat,
                       train_labels,
                       my_loss=best_loss,
                       my_penalty=best_pen,
                       my_alpha=best_alpha)
  # print mod.score(test_mat, test_labels)
  predicted_labels = mod.predict(test_mat)
  print 'Model Validation Score: ', mod.score(test_mat, test_labels)
  txt_labels = list(set(labels))
  cm = confusion_matrix(test_labels, predicted_labels, labels=txt_labels)
  '''
  labels, counts = iStats.get_labels()
  predicted_out = Counter(predicted_labels)
  print 'Total Labels', sum(counts.values()), len(labels)
  total_labels = len(labels)
  print ('{:<30}\t'*3).format('Label', 'Real Dist', 'Predicted Dist')
  for thing in sorted(counts, key= counts.get, reverse=True):
          params = [nn, loss, alpha, penalty]
    print ('{:<30}\t'*3).format(thing, counts[thing]*1./total_labels, \
        predicted_out[thing]*1.0/sum(predicted_out.values()))
  _, _, myHasher =
  load_sparse_csr(DATAPERSISTENCE)
  myHasher = myHasher.item()
  if False:
  while 1:
    data = tokenize_stem_stop(raw_input('Give me an example issue body\n'))
    example = myHasher.transform([data]) print mod.predict(example)
  '''
