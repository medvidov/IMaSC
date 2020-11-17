import json
import spacy


def prodigy_to_spacy(annotated_path: str):
    annotated_data = open(annotated_path)
    train_data = []
    for eg in annotated_data:
        j = json.loads(eg)
        if j['answer'] == 'accept':
            if 'spans' not in j:
                continue
            entities = [(span['start'], span['end'], span['label'])
                        for span in j['spans']]
            train_data.append((j['text'], {'entities': entities}))
    return train_data
