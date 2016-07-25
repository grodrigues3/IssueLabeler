
try:
  import matplotlib.pyplot as plt
except:
  pass

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
