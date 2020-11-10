import spacy
import json


class Metrics:
    def __init__(self, model_path = "IMaSC", annotated_path = "shaya_validate_test.jsonl", raw_path = "data/microwave_limb_sounder/validation_set.jsonl"):
        self.nlp = spacy.load(model_path)
        self.annotated_path = annotated_path
        self.raw_path = raw_path
        self.data_annotated = open(annotated_path)
        self.data_raw = open(raw_path)
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.truths = set()
        self.guesses = set()
        self.num_truths = 0
        self.accuracy = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.precision = 0.0

    def _reset_data(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.truths = set()
        self.guesses = set()
        self.num_truths = 0
        self.accuracy = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.precision = 0.0
        self.data_annotated = open(self.annotated_path)
        self.data_raw = open(self.raw_path)

    def _read_truths(self, label):
        #reads through annotated data and documents, pulls out "true" labels
        self.num_truths = 0
        for line in self.data_annotated:
            self.num_truths += 1
            j = json.loads(line)
            if 'spans' not in j:
                continue
            for span in j['spans']:
                if label == None or label == span["label"]:
                    self.truths.add((self.num_truths, span["start"], span["end"], span["label"]))


    def _make_guesses(self, label):
        #runs model on non-annotated data, stores model's guesses of labels
        if self.num_truths == 0:
            print("Has not read annotated data yet")
            return
        guess_line_num = 0
        for line in self.data_raw:
            guess_line_num += 1
            j = json.loads(line)
            doc = self.nlp(j["text"])
            for ent in doc.ents:
                if label == None or label == ent.label_:
                    self.guesses.add((guess_line_num, ent.start_char, ent.end_char, ent.label_))
            if guess_line_num == self.num_truths:
                break

    def _calc_tp(self):
        #calculates number of true positives
        self.tp += len(self.guesses.intersection(self.truths))

    def _calc_fp(self):
        #calculates number of false positives
        self.fp += len(self.guesses - self.truths)

    def _calc_fn(self):
        #calculates number of false negatives
        self.fn += len(self.truths - self.guesses)

    def _calc_recall(self):
        #calculates recall, a measure of true positives over predicted results
        self.recall = self.tp / (self.tp + self.fn)

    def _calc_precision(self):
        #calculates precision, a measure of true positives over total actual results
        self.precision = self.tp / (self.tp + self.fp)

    def _calc_f1(self):
        #calculates F1 score, "ultimate" measure of accuracy
        self.f1 = 2*(self.recall * self.precision) / (self.recall + self.precision)

    def calc_all_metrics(self):
        #after truths have been read and guesses have been made (self.truths and self.guesses are populated), calculates all metrics
        self._calc_tp()
        self._calc_fp()
        self._calc_fn()
        self._calc_recall()
        self._calc_precision()
        self._calc_f1()

    def display_metrics(self):
        #after all  metrics have been calculated, displays PRF
        print("Precision: ", round(self.precision, 2))
        print("Recall: ", round(self.recall, 2))
        print("F1: ", round(self.f1, 2))


    def calculate(self, label = None):
        #does "everything" - reads all data then calculates and displays metrics
        print("-------------------")
        if label == None:
            print("Displaying metrics for INSTRUMENT and SPACECRAFT")
        else:
            print("Displaying metrics for", label)
        self._read_truths(label)
        self._make_guesses(label)
        self.calc_all_metrics()
        self.display_metrics()
        self._reset_data()
        print("-------------------")


m1 = Metrics()
m1.calculate("SPACECRAFT")
m1.calculate("INSTRUMENT")
m1.calculate()