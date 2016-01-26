import math, collections

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    #Set all counts to 1 initially for Laplace Smoothing
    self.unigramCounts = collections.defaultdict(lambda: 1)
    self.bigramCounts = collections.defaultdict(lambda: 1)
    self.train(corpus)


  def train(self, corpus):
    for sentence in corpus.corpus:
      words = sentence.data
      for i in xrange(0, len(words)):
        word = words[i].word
        self.unigramCounts[word] += 1
        #Count bigram
        if (i >= 1):
            preword = words[i - 1].word
            bigram = "%s-%s" % (preword, word)
            self.bigramCounts[bigram] += 1
    pass

  def score(self, sentence):
    score = 0
    for i in xrange(0, len(sentence)):
        if (i >= 1):
            bigram = "%s-%s" % (sentence[i - 1], sentence[i])
            score += math.log(self.bigramCounts[bigram])
            preword = sentence[i - 1]
            score -= math.log(self.unigramCounts[preword] + len(self.bigramCounts))

    return score
