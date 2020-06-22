[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# IMaSC
Intelligent Mission and Scientific Instrument Classification. Applying unique NLP approaches to improve information extraction through scientific papers/Foundry A-Team Studies.

## The Data

Available datasets can be found in the `data` directory. The `microwave_limb_sounder` dataset contains a data dump of data from an Elasticsearch index, which contains documents with their parsed text ([PDF Miner](https://github.com/euske/pdfminer) was used to extract text from the PDF documents). The dataset also contains some, but not all, source PDFs. There are `1109` JSON documents but only `604` PDFs. The PDFs could be used with an altetrnative means of text extraction if desired and new machine-readable data generated for use in modeling. 

# Versioning
[Semantic versioning](https://semver.org/) is used for this project. If you are contributing to this project, please use semantic versioning guidelines when submitting your pull request.

# Contributing
Please use the issue tracker to report any unexpected behavior or desired features.

If you would like to contribute to development:
- Fork the repository.
- Create your changes in a branch corresponding to an open issue.
    - Bug fixes should be made in branches with the prefix `fix/`.
    - New capabilities or code improvements should be made in branches with the prefix `feature/`.
- Make a pull request into the repository's `dev` branch.
    - Pull requests should have the prefix [WIP] if they are works in progress.
    - Pull requests should have the prefix [MRG] if they are ready to merge.
- Upon successful completion of the pull request, it will be merged into `master`.

## Tests
When contributing, please run all existing unit tests. Add new tests as needed 
when adding new functionality. To run unit tests, use `pytest`:

```
python3 -m pytest --cov=IMaSC
```

# License
This project is licensed under the Apache 2.0 license.
