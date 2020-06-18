[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# IMaSC
Intelligent Mission and Scientific Instrument Classification. Applying unique NLP approaches to improve information extraction through scientific papers/Foundry A-Team Studies.

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
