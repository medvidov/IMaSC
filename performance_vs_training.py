from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import json
import imblearn
from utils import prodigy_to_spacy
from metrics_clean import Metrics
from tqdm import tqdm

class PerformanceVsTraining:
    def __init__(self, n:int = 20, verbose:bool = False, train_path: str = "training_annotations.jsonl", test_path: str = "shaya_validate_test.jsonl", label: list = ['INSTRUMENT', 'SPACECRAFT']) -> None:
        #starters from parameters
        self.train_path = train_path
        self.train_file = None
        self.test_path = test_path
        self.test_file = None
        self.num_data_points = n
        self.anns_per_point = None
        self.anns_this_round = 0 #changes with each round
        self.label = label
        self.metrics = Metrics("Baby", "shaya_validate_test.jsonl")
        self.t_vs_p = {}
        self.nlp = None
        self.verbose = verbose

    def _reset_data(self) -> None:
        #reset all metrics things between rounds
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
        self.data_annotated = open(self.annotated_path)
        self.data_raw = open(self.raw_path)

    def _prep_data(self) -> None:
        self.train_file = prodigy_to_spacy(self.train_path)
        num_anns = sum(1 for item in self.train_file) #total number of annotations
        self.train_file = prodigy_to_spacy(self.train_path)
        self.anns_per_point = num_anns / self.num_data_points
        self.test_file = prodigy_to_spacy(self.test_path)

    def _run_metrics(self) -> int:
        return self.metrics.calculate()


    def _train_one_round(self, i: int) -> None:
        n_iter = 100 #number of iterations. could make this customizable but I feel that it would be too messy
        #train model and save to self.nlp
        self.anns_this_round = i * self.anns_per_point
        if self.verbose:
            print("Training on %s annotations" % (self.anns_this_round))
        count = 0
        train_data = []
        for line in self.train_file:
            train_data.append(line)
            count += 1
            if count >= self.anns_this_round:
                break
        """Set up the pipeline and entity recognizer, and train the new entity."""
        random.seed(0)
        self.nlp = spacy.blank("en")  # create blank Language class
        # Add entity recognizer to model if it's not in the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe(ner)
        # otherwise, get it, so we can add labels to it
        else:
            ner = self.nlp.get_pipe("ner")

        for label in self.label:
            ner.add_label(label)  # add new entity label to entity recognizer
        optimizer = self.nlp.begin_training()

        move_names = list(ner.move_names)
        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]
        # only train NER
        with self.nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
            # show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module='spacy')

            sizes = compounding(1.0, 4.0, 1.001)
            # batch up the examples using spaCy's minibatch
            for itn in range(n_iter):
                random.shuffle(train_data)
                # Need some oversampling somewhere in here
                batches = minibatch(train_data, size=sizes)
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                #print("Losses", losses)
        output_dir = Path("Baby")
        if not output_dir.exists():
            output_dir.mkdir()
        self.nlp.meta["name"] = "BabyModel"  # rename model
        self.nlp.to_disk(output_dir)

    def run_test(self):
        self._prep_data()
        for i in tqdm(range(1, self.num_data_points + 1)):
            self._train_one_round(i)
            f1 = self._run_metrics()
            self.t_vs_p[round(self.anns_this_round,3)] = round(f1, 3)
            print(self.t_vs_p)


# @plac.annotations(
#     model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
#     new_model_name=("New model name for model meta.", "option", "nm", str),
#     output_dir=("Optional output directory", "option", "o", Path),
#     n_iter=("Number of training iterations", "option", "n", int),
# )


def main(model=None, new_model_name="imasc", output_dir="IMaSC", n_iter=100):
    p = PerformanceVsTraining(100, True)
    p.run_test()

if __name__ == "__main__":
    plac.call(main)
