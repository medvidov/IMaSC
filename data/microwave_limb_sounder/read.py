import json 


documents = list()

with open("mls_pubs.json", "rb") as f_in:
	data = f_in.readlines()
	for d in data:
		documents.append(json.loads(d))


print(len(documents))


