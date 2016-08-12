from collections import defaultdict
from csv import DictReader
from itertools import product


class IssueStats:

  def __init__(self):
    self.datafn = "./data/manyLabels.csv"
    self.labels = ['area', 'component', 'team']
    self.fieldnames = ['body', 'title'] + self.labels
    self.issueCnt = defaultdict(lambda: defaultdict(lambda:0))
    self.allCounts = defaultdict(lambda:0)
    #self.recently_deleted = defaultdict(lambda:0)
    curValidLabels = "./currentRouting.txt"
    with open(curValidLabels, 'r') as f:
      self.currentLabels = set(map(lambda x: x.strip(), f.readlines()))
    self._get_counts()

  def _get_counts(self):
    with open(self.datafn, 'r') as f:
      dr = DictReader(f, fieldnames= self.fieldnames)
      for i, line in enumerate(dr):
        for label in self.labels:
          theLabel = line[label]
          if theLabel in ["NULL", ""] or theLabel not in self.currentLabels:
            pass
          else: #good label
            curLabs = theLabel.split(";")
            for l in curLabs:
              if l in self.currentLabels:
                self.issueCnt[label][l] += 1
                self.allCounts[l] += 1
    self.totalIssueCnt = i+1


  def count_words(self, titles):
    """
    To estimate the size of the feature vector as passed into the hashing trick,
    count the vocabulary size (the number of unique tokens used in all titles and bodies)
    """
    myCounter = defaultdict(lambda: 0)
    for titleBody, label in self.get_training_examples():
      for word in titleBody.split():
        myCounter[word] +=1 
    return myCounter

  def get_corr(self):
    currentUsedLabels = filter(lambda x: x in self.currentLabels, self.allCounts.keys())
    myLabels = {k:i for i, k in enumerate(currentUsedLabels)}
    numLabels = len(myLabels)
    corMat = [[0]* numLabels for i in range(numLabels)] #numLabels x numLabels matrix
    with open(self.datafn, 'r') as f: 
      dr = DictReader(f, fieldnames= self.fieldnames)
      for i, line in enumerate(dr):
        lineLabels = []
        for label in self.labels:
          theLabel = line[label]
          if theLabel in ["NULL", ""]:
            pass
          elif ";" in theLabel:
            lineLabels += theLabel.split(";")
          elif theLabel in self.currentLabels:
            lineLabels += theLabel, 
        lineLabels = filter(lambda x: x in self.currentLabels, lineLabels)
        for a,b in product(lineLabels, lineLabels):
          x,y = myLabels[a], myLabels[b]
          corMat[x][y] +=1

    highest = []
    for i, row in enumerate(corMat):
      ind, val = max( ( (j,col) for j, col in enumerate(row) if j != i), key=lambda x:x[1])
      row_label = currentUsedLabels[i]
      highest.append((row_label, self.allCounts[row_label], currentUsedLabels[ind], val, val*100./self.allCounts[row_label]))
    print "Correlation Information"
    for fivel in sorted(highest, key=lambda x:x[-2], reverse=True):
      print "{:<30}\t{:3}\t{:<30}\t{:<3}\t{:2.2f}".format(*fivel)


  def print_counts(self, count_dict):
    print "Total Issues With Recently Deleted Labels: {} \
        Total Recently Deleted Labels {}".format(sum(count_dict.values()),
                                                 len(count_dict))
    for k in sorted(count_dict, key = count_dict.get, reverse=True):
      print "{:>20}\t{}\t{:.3f}".format(k, count_dict[k], count_dict[k]*1./self.totalIssueCnt)


  def get_labels(self, selector='team'):
    print "Loading Just The Labels..."
    counts = defaultdict(lambda:0)
    for i, (line, labels) in enumerate(self.get_training_examples(which_label=selector)):
      counts[labels] += 1
    return counts

  def get_training_examples(self, which_label=None):
    """
    Returns a generator of (lineBody (string), label (string)) pairs
    """
    print "Loading Titles and Labels To Be Used For Training Data..."
    with open(self.datafn, 'r') as f:
      dr = DictReader(f, fieldnames= self.fieldnames)
      for i, line in enumerate(dr):
        if which_label is not None and which_label in self.labels:
          if line[which_label] in ["NULL", ""]:
            continue
          #can have multiple team labels 
          retLabel = line[which_label]
          if ";" in retLabel: #more than one team
            retLabel = max(retLabel.split(";"), key = self.allCounts.get)
        else:
          #otherwise go through all labels and choose one thats used most often
          poss = []
          for label in self.labels:
            retLabel = line[label]
            if retLabel not in ["NULL", ""]:
              poss += retLabel, 
          if not poss:
            continue
          retLabel = max(poss, key = self.allCounts.get) #get the most commonly used label
        yield line['title'] + " " + line['body'],  retLabel

  def pretty_print(self):
    for broad_cat, spec_cat_count in self.issueCnt.iteritems():
      print broad_cat, sum(spec_cat_count.values())
      for label, cnt in spec_cat_count.iteritems():
        print "\t", label, cnt


  def print_top_n(self, n=10, by_sub_cat = False):
    if not by_sub_cat:
      #Overall top label usage
      flat_dict = {}
      print "Overall Top {} Labels".format(n)
      [flat_dict.update(self.issueCnt[label]) for label in self.labels]
      for top_area in sorted(flat_dict, key=flat_dict.get, reverse=True)[:n]:
        print "\t", top_area, flat_dict[top_area]
      print "-"*10
    else:
      #By sub category top label usage
      print "Top {} Labels By Subcategory".format(n)
      for label in self.labels:
        print "Broad Category: {0}".format(label)
        for top_area in sorted(self.issueCnt[label], key=self.issueCnt[label].get, reverse=True)[:n]:
          print "\t", top_area, self.issueCnt[label][top_area]
        print "-"*10

if __name__ == "__main__":
  iStats = IssueStats()
  #print iStats.get_labels()
  #iStats.print_top_n()
  #iStats.print_top_n(by_sub_cat=True)
  #iStats.print_counts(iStats.recently_deleted)
  #iStats.create_bar_chart()
  iStats.get_corr()




