from github import Github
from collections import Counter, defaultdict
import datetime, pdb

class K8s_Analyzer:

  def __init__(self):
    """
    """
    repo_name="kubernetes/kubernetes"
    password = raw_input("Enter your github oauth token")
    self.repo = Github("grodrigues3", password).get_repo(repo_name)
    self.routing_labels = ["area", "component", "team"]
    self.labels = []

  def get_labels(self):
    if not self.labels:
      self.labels = self.repo.get_labels()
    return self.labels

  def _get_label_names(self):
    """
    returns List[string] sorted label names
    """
    return sorted([label.name for label in self.get_labels()])

  def get_top_level_labels(self, write=False):
    """
    returns a dict mapping from broad category to list of all subcats
    e.g. area -> area/kubectl
    """
    broadToSub = defaultdict(list)
    for label in self.get_labels():
      high_level = label.name.split(r"/")[0]
      broadToSub[high_level] += label.name,
    if write:
      with open('cached_labels.txt', 'w') as f:
        for high_level in sorted(broadToSub, key= lambda x: len(broadToSub[x]), reverse=True):
          subLabels = broadToSub[high_level]
          f.write("{0} - {1} \n".format(high_level, len(subLabels)))
          for sub in subLabels:
              f.write("\t{}\n".format(sub))
    return broadToSub

  def extract_data(self):
    start_date = datetime.datetime(2016, 05, 01) #arbitrarily chosen for now
    issueCount = 0
    myFilter = lambda x: 'area' in x or 'component' in x or 'team' in x
    with open("training_data.txt", "w") as f, open("training_labels.txt", "w") as g, open("true_unlabeled.txt", 'w') as h:
      #get_issues returns issues and prs; don't want prs
      for i, issue in enumerate(self.repo.get_issues(since=start_date, state="closed")):
          #filter out the prs
          if not issue.pull_request:
            routing_labels = filter(myFilter, sorted([lab.name for lab in issue.labels]) ) 
            title = issue.title.strip("\t\n")
            if not routing_labels:
              try:
                h.write("{}, {}\n".format(issue.number, title)) #, body.strip()) )
              except:
                pass
            else:
              try:
                f.write("{}, {}\n".format(issue.number, title))
                g.write(("{}, "*(len(routing_labels)-1) +"{}" + "\n").format(*routing_labels))
                issueCount += 1
                if issueCount % 20 == 0:
                  print issueCount, issue.number, title, routing_labels,
              except:
                continue

if __name__ == "__main__":
  TestK8s = K8s_Analyzer()
  """
  label_count = TestK8s.get_top_level_labels()
  row_format = "{:>30}"*2
  for key in sorted(label_count, key = label_count.get, reverse=True):
    print row_format.format(key, label_count[key])
  """
  TestK8s.extract_data()
