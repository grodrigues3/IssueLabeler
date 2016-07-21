from collections import defaultdict
from csv import DictReader
from itertools import product
import pdb
try:
  import matplotlib.pyplot as plt
except:
  pass

curValidLabels = "./currentRouting.txt"

class IssueStats:
  def __init__(self):
    self.datafn = "./data/raw_data.csv"
    self.labels = ['component', 'team', 'area']
    self.fieldnames = ['body', 'title'] + self.labels
    self.issueCnt = defaultdict(lambda: defaultdict(lambda:0))
    self.allCounts = defaultdict(lambda:0)
    self.recently_deleted = defaultdict(lambda:0)
    with open(curValidLabels, 'r') as f:
      self.currentLabels = set(map(lambda x: x.strip(), f.readlines()))
    self._get_counts()

  def _get_counts(self):
    with open(self.datafn, 'r') as f:
      dr = DictReader(f, fieldnames= self.fieldnames)
      for i, line in enumerate(dr):
        for label in self.labels:
          theLabel = line[label]
          if theLabel in ["NULL", ""]:
            pass
          elif theLabel not in self.currentLabels:
            self.recently_deleted[theLabel] += 1
          else: #good label
            self.issueCnt[label][theLabel] += 1
            self.allCounts[theLabel] += 1
    self.totalIssueCnt = i+1


  def get_corr(self):
    currentLabels = filter(lambda x: x in self.currentLabels, self.allCounts.keys())
    myLabels = {k:i for i, k in enumerate(currentLabels)}
    numLabels = len(myLabels)
    corMat = [ [0]* numLabels for i in range(numLabels)] #numLabels x numLabels matrix
    with open(self.datafn, 'r') as f:
      dr = DictReader(f, fieldnames= self.fieldnames)
      for i, line in enumerate(dr):
        lineLabels = []
        for label in self.labels:
          theLabel = line[label]
          if theLabel in ["NULL", ""]:
            pass
          elif theLabel in self.currentLabels:
            lineLabels += theLabel, 
        for a,b in product(lineLabels, lineLabels):
          x,y = myLabels[a], myLabels[b]
          corMat[x][y] +=1 

    highest = []
    for i, row in enumerate(corMat):
      ind, val = max( ( (j,col) for j, col in enumerate(row) if j != i), key=lambda x:x[1])
      row_label = currentLabels[i]
      highest.append((row_label, self.allCounts[row_label], currentLabels[ind], val, val*100./self.allCounts[row_label]))
    print "Correlation Information"
    for fivel in sorted(highest, key=lambda x:x[-2], reverse=True):
      print "{:<30}\t{:3}\t{:<30}\t{:<3}\t{:2.2f}".format(*fivel)


  def print_counts(self, count_dict):
    print "Total Issues With Recently Deleted Labels: {} \
        Total Recently Deleted Labels {}".format(sum(count_dict.values()),
                                                 len(count_dict))
    for k in sorted(count_dict, key = count_dict.get, reverse=True):
      print "{:>20}\t{}\t{:.3f}".format(k, count_dict[k], count_dict[k]*1./self.totalIssueCnt)



  def get_training_examples(self):
    with open(self.datafn, 'r') as f:
      dr = DictReader(f, fieldnames= self.fieldnames)
      for i, line in enumerate(dr):
        poss_labels = []
        # right now, enforce a single label 
        # if there's a component use it, otherwise team, otherwise area
        for label in self.labels:
          poss = []
          if line[label] not in ["NULL", ""]:
            poss += (self.issueCnt[label][line[label]], line[label]),
          if not poss:
            continue
          best_label = max(poss, key = self.issueCnt.get)[1] #get the most commonly used label
          yield line['title'] + " " + line['body'],  best_label

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

  def create_bar_chart(self):
    threshold = .002
    counts = []
    labels = []
    sorted_keys = list(sorted(self.allCounts, key = self.allCounts.get, reverse=True))
    for i, curLabel in enumerate(sorted_keys):
      labels += curLabel,
      counts += self.allCounts[curLabel],
      print curLabel, counts[-1]
      if counts[-1]*1./self.totalIssueCnt < threshold:
        break
    fig, ax = plt.subplots()
    width = .02
    x_ticks = [x+width for x in range(len(labels))]
    rects = ax.bar(x_ticks, counts)
    for i, rect in enumerate(rects):
      ax.text(rect.get_x() + rect.get_width()/2., 1.023*counts[i],
              '{}\n{:2.1f}%'.format(counts[i], counts[i]*100./self.totalIssueCnt),
                    ha='center', 
                    va='bottom')
    ax.set_ylabel("Number of Issues Assigned")
    plt.xticks([thing+2*width for thing in x_ticks], labels, rotation=90)
    plt.title("Out of {} Issues Top {} Labels By Usage".format(self.totalIssueCnt, len(labels)), fontsize= 18)
    plt.text(x_ticks[-1],counts[3],"*{} Total Labels".format(len(sorted_keys)))
    plt.show()

  def plot_confusion_matrix(self, cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = range(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
if __name__ == "__main__":
  iStats = IssueStats()
  iStats.print_top_n()
  iStats.print_top_n(by_sub_cat=True)
  iStats.print_counts(iStats.recently_deleted)
  #iStats.create_bar_chart()
  iStats.get_corr()




