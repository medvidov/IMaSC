[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# IMaSC

Intelligent Mission and Scientific Instrument Classification. Applying unique NLP approaches to improve information extraction through scientific papers/Foundry A-Team Studies.

## The Data

Available datasets can be found in the `data` directory. The `microwave_limb_sounder` dataset contains a data dump of data from an Elasticsearch index, which contains documents with their parsed text ([PDFMiner](https://github.com/euske/pdfminer) was used to extract text from the PDF documents). The dataset also contains some, but not all, source PDFs. There are `1109` JSON documents but only `604` PDFs. The PDFs could be used with an altetrnative means of text extraction if desired and new machine-readable data generated for use in modeling.

### Generating datasets

To generate training, validation, and testing sets, run `parser.py` with default inputs. This will generate the three files `training_set.jsonl` , `validation_set.jsonl` , and `testing_set.jsonl` in the 
`data/microwave_limb_sounder` directory.

### Using Prodigy

Prodigy will allow you to annotate your datasets. Please note that my Prodigy wheel installation path is specific to my laptop in `requirements.txt` at this time.

* To start Prodigy, run `prodigy ner.manual name_of_dataset name_of_model ./path/to/dataset.jsonl --label INSTRUMENT,SPACECRAFT` in the `IMaSC` directory. 
    - In my case I ran `prodigy ner.manual train_imasc en_core_web_sm ./data/microwave_limb_sounder/training_set.jsonl --label INSTRUMENT,SPACECRAFT` .
* From there, open a browser and enter [http://localhost:8080/](http://localhost:8080/) in 

the search bar. Prodigy should be running on port 8080 by default.

* To annotate text, click on the annotation you wish to apply and highlight the text you wish to annotate. 

Prodigy will automatically apply the annotation. 

* To remove an annotation, hover over the top left corner of an existing annotation 

and click the "X".

* Once you have finished annotating a piece of text (you may not need to annotate anything), 
 
click the green check mark. If a piece of text is not appropriate for annotation, 
click the grey no symbol to skip it.

* You can also click the grey return to undo an annotation.

* To export annotations as a JSONL, run `prodigy db-out train_imasc > ./annotations.jsonl`.

#### What to label

Currently, IMaSC supports labeling of scientific instruments (i.e. `MLS` ) and the spacecraft (i.e. `Aura satellite` ) that carry them. 
Using the directions above, label all instances of scientific instruments and spacecraft in the text.

#### Handling Uncertainties While Annotating

While reading through data, look out for acronyms or things that have “Satellite” or something similar for spacecraft, or words that end in “meter” or similar for instruments.

When you identify a potential token, reference the annotations.md file and see if it’s already in there. If the term is in the document as a spacecraft or instrument, annotate it accordingly. If it’s in the document as something else (“model” or “other”), ignore it.

If you think you’ve found a token but it isn’t in annotations.md, try to classify it on your own:
* Use context. Sentences like “aboard the XXXX” or “on the YYYY” might indicate spacecraft, while “measurements taken by ZZZZ” might indicate instruments
* Often times, terms expressed in the form “ XXX / YYY” like “UARS / MLS” are of the form “ SPACECRAFT / INSTRUMENT “ 
* Google the term along with with “NASA”, like “MLS NASA” and read into several articles to see what entity the term might be associated with
* If you’re really unsure, it’s best to note it somewhere and see if it comes up again; compare contexts it appears in
* If you can find information but the information is ambiguous, like you can’t tell if something is an experiment or an instrument (ex. HALOE), just pick a label and stay consistent with it


#### Training

Train the model with the following command: `prodigy ner.batch-train train_imasc en_core_web_sm -n 100`. To train a model with only one entity type run `prodigy ner.batch-train train_imasc en_core_web_sm -n 100 -l ENTITY`. 
A flowchart for how to train your specific model can be found [here](https://prodi.gy/prodigy_flowchart_ner-36f76cffd9cb4ef653a21ee78659d366.pdf). About 4000 annotations are needed to train the model.

## The API (Application Programming Interface)
Included in this repository is a basic API that runs the model on user input data and displays a list of tokens the model found along with coverage data. To use the API, run “python api.py” from the api_stuff directory in the terminal. Your command line should spit out a link you can then use to access the API in your default browser. 

In your browser, either enter text or drop a PDF (a recommended PDF is provided in this repository) and click “submit.”


## Versioning

[Semantic versioning](https://semver.org/) is used for this project. If you are contributing to this project, please use semantic versioning guidelines when submitting your pull request.

## Contributing

Please use the issue tracker to report any unexpected behavior or desired features.

If you would like to contribute to development:

* Fork the repository.
* Create your changes in a branch corresponding to an open issue.
    - Bug fixes should be made in branches with the prefix `fix/` .
    - New capabilities or code improvements should be made in branches with the prefix `feature/` .
* Make a pull request into the repository's `dev` branch.
    - Pull requests should have the prefix [WIP] if they are works in progress.
    - Pull requests should have the prefix [MRG] if they are ready to merge.
* Upon successful completion of the pull request, it will be merged into `master` .

### Tests

When contributing, please run all existing unit tests. Add new tests as needed 
when adding new functionality. To run unit tests, use `pytest` :

``` 
python3 -m pytest --cov=IMaSC
```

## License

This project is licensed under the Apache 2.0 license.
