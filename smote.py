import spacy
import json

nlp = spacy.load("IMaSC")
#load up annotated data
#count number of instruments vs number of spacecraft
#decide on which one we need to oversample and how much
#duplicate random samples of the lower one until they're even
#output into another file??? or use same file???

train_data = open("11_9_20_annos.jsonl", "r+")
inst_space_count = {"INSTRUMENT": 0, "SPACECRAFT": 0}
for line in train_data:
    j = json.loads(line)
    if 'spans' not in j:
        continue
    for span in j['spans']:
        print(span)

to_add = inst_space_count["INSTRUMENT"] - inst_space_count["SPACECRAFT"]
to_add = 1
#TARGLAB = min(inst_space_count, key=inst_space_count.get) #target label to be oversampled
#duplicate samples
count = 0
print("here1")
#train_data.close()
train_data_2 = open("11_9_20_annos.jsonl", "r+")
train_data_j = json.load(train_data_2)
for line in train_data_j:
    print("here2")
    #json.dump(line, train_data)
    print("here4")
print("done")

#put into a new file
