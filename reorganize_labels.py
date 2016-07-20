import pdb
newFn = "newMap.txt"
def get_new():
  d = {}
  with open('./data/old_new_map.txt', 'r') as f:
    oldLabels = [line.strip().split(",")[0] for line in f]
  import pdb
  newLabels = [
    "area/networking",
    "area/storage",
    "area/security",
    "area/autoscaling",
    "area/federation",
    "area/qoS",
    "area/node",
    "area/performance",
    "area/kubelet",
    "area/clientlib",
    "area/kube-proxy",
    "area/controller-manager",
    "area/apiserver",
    "area/ui",
    "area/kubectl",
    "area/nodeController",
    "area/scheduler",
    "area/node-e2e",
    "area/minikube",
    "area/deployment",
    "area/dns"
  ]

  newLabels = {newLabels.index(l):l for l in newLabels}
  newLabels[len(newLabels)] = "NOT SURE"
  newMap = {}
  for i in sorted(newLabels):
    print i, newLabels[i]


  print "please provide your relabels"
  for i, thing in enumerate(oldLabels):
    new = raw_input(str(i) + "\t" + thing +"\n\t")
    try:
      ind = int(new)
      newMap[thing] = newLabels[ind]
    except:
      newMap[thing] = new



def check_new():
  c = 0
  d = {}
  with open(newFn, 'r') as f:
    for i, line in enumerate(f):
      old, new = line.strip().split(",")
      if not new or new == "NOT SURE":
        new = raw_input(str(i) + "\t" + old +"\n\t")
      d[old] = new
  return d


if __name__ == "__main__":
  this = check_new()
