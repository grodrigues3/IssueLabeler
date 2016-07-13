
from collections import defaultdict
from csv import DictReader
import pdb

class IssueStats:
  def __init__(self):
    self.datafn = "./data/raw_data.csv"
    self.labels = ['component', 'team', 'area']
    self.fieldnames = ['body', 'title'] + self.labels
    self.issueCnt = self._get_counts()
    self.totalIssueCnt = 0

  def _get_counts(self):
    issueCnt = defaultdict(lambda: defaultdict(lambda:0))
    with open(self.datafn, 'r') as f:
      dr = DictReader(f, fieldnames= self.fieldnames)
      for i, line in enumerate(dr):
        for label in self.labels:
          if line[label] not in ["NULL", ""]:
            issueCnt[label][line[label]] += 1
    self.totalIssueCnt = i+1
    return issueCnt

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

if __name__ == "__main__":
  iStats = IssueStats()
  iStats.print_top_n()
  iStats.print_top_n(by_sub_cat=True)




