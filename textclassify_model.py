import sys
from collections import Counter
import math
import numpy as np

"""
Arinjay Jain
File comment here:
"""

"""
Cite your sources here:
- For positive and negative lexicon:
    1. Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."
        Proceedings of the ACM SIGKDD International Conference on Knowledge 
       Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
       Washington, USA, 
    2. Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing 
       and Comparing Opinions on the Web." Proceedings of the 14th 
       International World Wide Web conference (WWW-2005), May 10-14, 
       2005, Chiba, Japan.

"""


def generate_tuples_from_file(training_file_path):
    """
  Generates tuples from file formatted like:
  id\ttext\tlabel
  Parameters:
    training_file_path - str path to file to read in
  Return:
    a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
  """
    f = open(training_file_path, "r", encoding="utf8")
    listOfExamples = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.split("\t")
        for i in range(len(dataInReview)):
            # remove any extraneous whitespace
            dataInReview[i] = dataInReview[i].strip()
        t = tuple(dataInReview)
        listOfExamples.append(t)
    f.close()
    return listOfExamples


def load_word_list(filename):
    """
    Loads a lexicon from a plain text file in the format of one word per line.
    Parameters:
    filename (str): path to file

    Returns:
    list: list of words
    """
    with open(filename, 'r', encoding="utf-8") as f:
        # skip the header content
        for line in f:
            if line.strip() == "":
                break
        # read the rest of the lines into a list
        return [line.strip() for line in f]


def find_confusion_matrix(gold_labels, predicted_labels):
    """
    Finds the confusion matrix for the given gold labels and predicted labels, that is computes
    the true positives, true negatives, false positives and false negatives.
    Args:
        gold_labels: "Truth" hand labels
        predicted_labels: Labels predicted by the model

    Returns:
    true positive, true negative, false positive and false negative values
    """
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for actual, predicted in zip(gold_labels, predicted_labels):
        if predicted == "1":
            if actual == "1":
                true_positive += 1
            else:
                false_positive += 1
        if predicted == "0":
            if actual == "0":
                true_negative += 1
            else:
                false_negative += 1
    return true_positive, false_positive, true_negative, false_negative


def precision(gold_labels, predicted_labels):
    """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """
    true_positive, false_positive, true_negative, false_negative = find_confusion_matrix(gold_labels, predicted_labels)
    precision_score = true_positive / (true_positive + false_positive)
    return precision_score


def recall(gold_labels, predicted_labels):
    """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """
    true_positive, false_positive, true_negative, false_negative = find_confusion_matrix(gold_labels, predicted_labels)
    recall_score = true_positive / (true_positive + false_negative)
    return recall_score


def f1(gold_labels, predicted_labels):
    """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """
    true_positive, false_positive, true_negative, false_negative = find_confusion_matrix(gold_labels, predicted_labels)
    precision_score = true_positive / (true_positive + false_positive)
    recall_score = true_positive / (true_positive + false_negative)
    return 0 if precision_score + recall_score == 0 else (2 * precision_score * recall_score) / (precision_score + recall_score)


"""
Implement any other non-required functions here
"""

"""
implement your TextClassify class here
"""


class TextClassify:

    def __init__(self):
        self.labels = {}
        self.vocab = set()
        self.word_counts = {}

    def train(self, examples):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """
        for review_id, review, label in examples:
            if label not in self.labels:
                self.labels[label] = 0
                self.word_counts[label] = {}
            self.labels[label] += 1
            for word in review.split():
                self.vocab.add(word)
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 0
                self.word_counts[label][word] += 1

    def score(self, data):
        """
         Score a given piece of text
         you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here

         Parameters:
           data - str like "I loved the hotel"
         Return: dict of class: score mappings
         return a dictionary of the values of P(data | c)  for each class,
         as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
         """
        # P(Word|Class) = (Count of word in class + 1) / (Number of words in class + Vocabulary size)
        # If word is not in class P(Word|Class) = 1
        scores = {}
        words = data.split()
        for label in self.labels:
            score = 0
            denominator = sum(self.word_counts[label].values()) + len(self.vocab)
            for word in words:
                if word not in self.vocab:
                    continue
                elif word in self.word_counts[label]:
                    score += np.log((self.word_counts[label][word] + 1) / denominator)
                else:
                    score += np.log(1 / denominator)
            scores[label] = math.exp(score + np.log(self.labels[label] / sum(self.labels.values())))
        return scores

    def classify(self, data):
        """
        Label a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: string class label
        """
        scores = self.score(data)
        classified_labels = []
        max_score = max(scores.values())
        for label, score in scores.items():
            if score == max_score:
                classified_labels.append(label)
        if len(classified_labels) > 1:
            return min(classified_labels)
        return classified_labels[0]

    def featurize(self, data):
        """
        We use this format to make implementation of your TextClassifyImproved model more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """
        return [(word, True) for word in data.split()]

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:

    def __init__(self):
        self.pos_lex = load_word_list("positive_words.txt")
        self.neg_lex = load_word_list("negative_words.txt")
        self.labels = {}
        self.vocab = set()
        self.word_counts = {}

    def train(self, examples):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """
        pass

    def score(self, data):
        """
        Score a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: dict of class: score mappings
        """
        pass

    def classify(self, data):
        """
        Label a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: string class label
        """
        pass

    def featurize(self, data):
        """
        We use this format to make implementation of this class more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """
        pass

    def __str__(self):
        return "Logistic Regression"


def main():
    training = sys.argv[1]
    testing = sys.argv[2]
    training_file = open(training)
    testing_file = open(testing)

    classifier = TextClassify()
    print(classifier)
    # do the things that you need to with your base class

    # report precision, recall, f1

    improved = TextClassifyImproved()
    print(improved)
    # do the things that you need to with your improved class

    # report final precision, recall, f1 (for your best model)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
        sys.exit(1)

    main()
