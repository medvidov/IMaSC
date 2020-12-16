import spacy
import json
from collections import defaultdict
import pprint

# Number of alternate analyses to consider. More is slower, and not necessarily better -- you need to experiment on your problem.
beam_width = 16
# This clips solutions at each step. We multiply the score of the top-ranked action by this value, and use the result as a threshold. This prevents the parser from exploring options that look very unlikely, saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
beam_density = 0.0001
nlp = spacy.load('IMaSC')
pp = pprint.PrettyPrinter(indent=4)

data = open("data/microwave_limb_sounder/validation_set.jsonl")
texts = []
n = 0
for line in data:
    if n == 100: break
    j = json.loads(line)
    texts.append(j["text"])
    n += 1

docs = list(nlp.pipe(texts, disable=['ner']))


#docs = list(nlp(texts))
beams = nlp.entity.beam_parse(docs, beam_width=beam_width, beam_density=beam_density)

entity_scores = set()
for doc, beam in zip(docs, beams):
    for score, ents in nlp.entity.moves.get_beam_parses(beam):
        for ent in ents:
            entity_scores.add((doc[ent[0]:ent[1]], ent[2], score)) #the text, the label, and the score

pp.pprint((entity_scores))
