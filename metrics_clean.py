import spacy
import json


class Metrics:
    def __init__(self, model_path = "IMaSC", annotated_path = "shaya_validate_test.jsonl", raw_path = "data/microwave_limb_sounder/validation_set.jsonl"):
        self.nlp = spacy.load(model_path)
        self.data_annotated = open(annotated_path)
        self.data_raw = open(raw_path)
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.truths = set()
        self.guesses = set()
        self.numTruths = 0
        self.accuracy = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.precision = 0.0

    def readTruths(self, label):
        #reads through annotated data and documents
        self.numTruths = 0
        for line in self.data_annotated:
            self.numTruths += 1
            j = json.loads(line)
            if 'spans' not in j:
                continue
            for span in j['spans']:
                if label == None or label == span["label"]:
                    self.truths.add((self.numTruths, span["start"], span["end"], span["label"]))

    def makeGuesses(self, label):
        if self.numTruths == 0:
            print("Has not read annotated data yet")
            return
        guessLineNum = 0
        for line in self.data_raw:
            guessLineNum += 1
            j = json.loads(line)
            doc = self.nlp(j["text"])
            for ent in doc.ents:
                if label == None or label == ent.label_:
                    self.guesses.add((guessLineNum, ent.start_char, ent.end_char, ent.label_))
            if guessLineNum == self.numTruths:
                break

    def calcTp(self):
        self.tp += len(self.guesses.intersection(self.truths))

    def calcFp(self):
        self.fp += len(self.guesses - self.truths)

    def calcFn(self):
        self.fn += len(self.truths - self.guesses)

    def calcRecall(self):
        self.recall = self.tp / (self.tp + self.fn)

    def calcPrecision(self):
        self.precision = self.tp / (self.tp + self.fp)

    def calcF1(self):
        self.f1 = 2*(self.recall * self.precision) / (self.recall + self.precision)

    def calcAllMetrics(self):
        self.calcTp()
        self.calcFp()
        self.calcFn()
        self.calcRecall()
        self.calcPrecision()
        self.calcF1()

    def displayMetrics(self):
        print("Precision: ", round(self.precision, 2))
        print("Recall: ", round(self.recall, 2))
        print("F1: ", round(self.f1, 2))


    def doEverything(self, label = None):
        print("-------------------")
        if label == None:
            print("Displaying metrics for INSTRUMENT and SPACECRAFT")
        else:
            print("Displaying metrics for", label)
        self.readTruths(label)
        self.makeGuesses(label)
        self.calcAllMetrics()
        self.displayMetrics()
        print("-------------------")


m1 = Metrics()
m2 = Metrics()
m3 = Metrics()
m1.doEverything("SPACECRAFT")
m2.doEverything("INSTRUMENT")
m3.doEverything()
