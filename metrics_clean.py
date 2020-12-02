import spacy
import json
from utils import prodigy_to_spacy

class Metrics:
    def __init__(self, model_path: str = "IMaSC", annotated_path: str = "shaya_validate_test.jsonl")-> None:
        self.model_path = model_path
        self.nlp = spacy.load(model_path)
        self.annotated_path = annotated_path
        self.data_annotated = prodigy_to_spacy(annotated_path)
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

    def _reset_data(self) -> None:
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
        self.data_annotated = prodigy_to_spacy(self.annotated_path)
        self.nlp = spacy.load(self.model_path)

    def _read_truths(self, label: str)  -> None:
        #reads through annotated data and documents, pulls out "true" labels
        self.num_truths = 0
        for line in self.data_annotated:
            self.num_truths += 1
            j = line
            #j = json.loads(line)
            if 'entities' not in j[1]:
                continue
            for ent in j[1]['entities']:
                if label == None or label == ent[2]:
                    #print(j[0][ent[0]:ent[1]], ent[0], ent[1], ent[2])
                    self.truths.add((self.num_truths, ent[0], ent[1], ent[2]))

    def _make_guesses(self, label: str)  -> None:
        #runs model on non-annotated data, stores model's guesses of labels
        if self.num_truths == 0:
            print("Has not read annotated data yet")
            return
        guess_line_num = 0
        for line in self.data_annotated:
            guess_line_num += 1
            j = line
            doc = self.nlp(j[0])
            for ent in doc.ents:
                if label == None or label == ent.label_:
                    #print(ent.text, ent.start_char, ent.end_char, ent.label_)
                    self.guesses.add((guess_line_num, ent.start_char, ent.end_char, ent.label_))
            if guess_line_num == self.num_truths:
                break

    def _calc_tp(self) -> None:
        #calculates number of true positives
        self.tp += len(self.guesses.intersection(self.truths))

    def _calc_fp(self) -> None:
        #calculates number of false positives
        self.fp += len(self.guesses - self.truths)

    def _calc_fn(self) -> None:
        #calculates number of false negatives
        self.fn += len(self.truths - self.guesses)

    def _calc_recall(self) -> None:
        #calculates recall, a measure of true positives over predicted results
        self.recall = self.tp / (self.tp + self.fn)

    def _calc_precision(self) -> None:
        #calculates precision, a measure of true positives over total actual results
        self.precision = self.tp / (self.tp + self.fp)

    def _calc_f1(self) -> None:
        #calculates F1 score, "ultimate" measure of accuracy
        self.f1 = 2*(self.recall * self.precision) / (self.recall + self.precision)

    def calc_all_metrics(self) -> None:
        #after truths have been read and guesses have been made (self.truths and self.guesses are populated), calculates all metrics
        self._calc_tp()
        self._calc_fp()
        self._calc_fn()
        self._calc_recall()
        self._calc_precision()
        self._calc_f1()


    def display_metrics(self) -> None:
        #after all  metrics have been calculated, displays PRF
        print("Precision: ", round(self.precision, 2))
        print("Recall: ", round(self.recall, 2))
        print("F1: ", round(self.f1, 2))


    def calculate(self, label: str = None) -> int:
        #does "everything" - reads all data then calculates and displays metrics
        print("-------------------")
        if label == None:
            print("Displaying metrics for INSTRUMENT and SPACECRAFT")
        else:
            print("Displaying metrics for", label)
        self._read_truths(label)
        print("-------------")
        self._make_guesses(label)
        self.calc_all_metrics()
        result = self.f1
        self._reset_data()
        print("-------------------")
        return result
