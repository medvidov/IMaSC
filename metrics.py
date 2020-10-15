import spacy
import json

nlp = spacy.load("/Users/shaya/Desktop/Git/shayanaimi/IMaSC/IMaSC")
# doc = nlp("/Users/shaya/Desktop/Git/shayanaimi/IMaSC/data/microwave_limb_sounder/validation_set.jsonl")
non_annotated_source = open("/Users/shaya/Desktop/Git/shayanaimi/IMaSC/data/microwave_limb_sounder/validation_set.jsonl")
annotated_source = open("/Users/shaya/Desktop/Git/shayanaimi/IMaSC/shaya_validate_test.jsonl")
guesses = set()
truths = set()
lineNum = 0
for line in annotated_source:
    lineNum += 1
    j = json.loads(line)
    if 'spans' not in j:
        continue
    for span in j['spans']:
        truths.add((lineNum, span["start"], span["end"], span["label"]))
guessLineNum = 0
for line in non_annotated_source:
    guessLineNum += 1
    j = json.loads(line)
    doc = nlp(j["text"])
    for ent in doc.ents:
        guesses.add((guessLineNum, ent.start_char, ent.end_char, ent.label_))
    if guessLineNum == lineNum:
        break
tp = 0.0
fp = 0.0
fn = 0.0
tp += len(guesses.intersection(truths))
fn += len(truths - guesses)
fp += len(guesses - truths)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = 2*(recall * precision) / (recall + precision)
print("precision: ", round(precision, 2))
print("recall: ", round(recall, 2))
print("f1: ", round(f1, 2))
