import json
import spacy

class Metrics:
    def __init__(self, model_path: str = "IMaSC", raw_path = "data/microwave_limb_sounder/validation_set.jsonl", text_key = "text", n = 10) -> None:
        self.nlp = spacy.load(model_path)
        self.raw_path = raw_path
        self.data_raw = open(raw_path)
        self.guesses = set()
        self.text_key = text_key
        self.n = n #max number of guesses you want made

    def _reset_data(self):
        self.guesses = set()
        self.data_raw = open(self.raw_path)

    def _make_guesses(self, label):
        #runs model on non-annotated data, stores model's guesses of labels
        guesscount = 0
        for line in self.data_raw:
            if guesscount >= self.n:
                break
            j = json.loads(line)
            if "_source" not in j:
                continue
            if "technical_abstract" not in j["_source"]:
                continue
            doc = self.nlp(j["_source"]["technical_abstract"])
            for ent in doc.ents:
                if label == None or label == ent.label_:
                    self.guesses.add((ent.text, ent.label_))
            guesscount += 1

    def _display_results(self):
        #display all results
        for guess in self.guesses:
            print(guess)
        pass

    def calculate(self, label = None):
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


m1 = Metrics("IMaSC", "sbir.json", "technical_abstract", 100)
m1.calculate()
