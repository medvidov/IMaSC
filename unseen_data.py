import json
import spacy
from utils import prodigy_to_spacy

class Metrics:
    def __init__(self, model_path: str = "IMaSC", raw_path: str = "data/microwave_limb_sounder/validation_set.jsonl", num_guesses: int = 100, prim_key: str = "text", sec_key: str = None) -> None:
        self.nlp = spacy.load(model_path)
        self.raw_path = raw_path
        self.data_raw = open(raw_path)
        self.guesses = set()
        self.prim_key = prim_key
        self.sec_key = sec_key
        self.n = num_guesses #max number of guesses you want made

    def _reset_data(self) -> None:
        self.guesses = set()
        self.data_raw = open(self.raw_path)

    def _make_guesses(self, label: str) -> None:
        #runs model on non-annotated data, stores model's guesses of labels
        guesscount = 0
        for line in self.data_raw:
            if guesscount >= self.n:
                break
            j = json.loads(line)
            """
            if primary key doesn't exist in j continue
            if secondary key is not none and secondary key doesn't exist in j[primary key] continue
            if secondary key is none, then nlp on the text in primary key
            if secondary key is not none, then nlp on j[primary][secondary]
            """
            if self.prim_key not in j:
                continue
            if self.sec_key != None:
                if self.sec_key not in j[self.prim_key]:
                    print('here')
                    continue
                doc = self.nlp(j[self.prim_key][self.sec_key])
            elif self.sec_key == None:
                doc = self.nlp(j[self.prim_key])
            else:
                print("something is wrong")
            for ent in doc.ents:
                if label == None or label == ent.label_:
                    self.guesses.add((ent.text, ent.label_))
            guesscount += 1

    def _display_results(self) -> None:
        #display all results
        for guess in self.guesses:
            print(guess)
        pass

    def calculate(self, label:str  = None) -> None:
        #does "everything" - reads all data then calculates and displays metrics
        print("-------------------")
        self._make_guesses(label)
        if label == None:
            print("Displaying result for INSTRUMENT and SPACECRAFT")
        else:
            print("Displaying metrics for", label)
        self._display_results()
        self._reset_data()
        print("-------------------")


m1 = Metrics("IMaSC", "training_annotations.jsonl", 100, "text")
m1.calculate()
