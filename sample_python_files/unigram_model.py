import math, collections

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramLapCounts = collections.defaultdict(lambda: 1)
    self.total = 0
    self.train(corpus)

  def train(self, corpus): 
    for sentence in corpus.corpus:
      for datum in sentence.data:  
        word = datum.word
        self.unigramLapCounts[word] += 1
        self.total += 1
    self.total += len(self.unigramLapCounts) #Correction needed given Laplace smoothing
    pass

  def score(self, sentence):
   score = 0.0 
    for token in sentence:
      count = self.unigramLapCounts[token]
      score += math.log(count)
      score -= math.log(self.total)
    return score
