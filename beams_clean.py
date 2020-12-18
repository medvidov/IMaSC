import spacy
import json
from collections import defaultdict
import pprint

class Beams:
    def __init__(self, model_path:str = "IMaSC", data_path: str = "data/microwave_limb_sounder/validation_set.jsonl", data_list: list = None, n:int = 100)->None:
        self.beam_width = 16
        self.beam_density = .0001
        self.nlp = spacy.load(model_path)
        self.data = open(data_path)
        self.texts = []
        self.n = n
        self.entity_scores = set()
        count = 0
        if data_list == None:
            for line in self.data:
                if count == self.n: break
                j = json.loads(line)
                self.texts.append(j["text"])
                count += 1
        else:
            self.texts = data_list

    def perform(self)->None:
        docs = list(self.nlp.pipe(self.texts, disable=['ner']))
        beams = self.nlp.entity.beam_parse(docs, beam_width=self.beam_width, beam_density=self.beam_density)
        for doc, beam in zip(docs, beams):
            for score, ents in self.nlp.entity.moves.get_beam_parses(beam):
                for ent in ents:
                    self.entity_scores.add((doc[ent[0]:ent[1]], ent[2], score)) #the text, the label, and the score


    def get_scores(self)->set:
        return self.entity_scores

    def _reset(self)->None:
        self.entity_scores = set()

    def display(self)->None:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint((self.entity_scores))
        pass

b = Beams()
b.perform()
b.get_scores()
b.display()
