# JProjectL
Name is a work in progress. Applying unique NLP approaches to improve information extraction through scientific papers/Foundry A-Team Studies.


## The Data

Available datasets can be found in the `data` directory. The `microwave_limb_sounder` dataset contains a data dump of data from an Elasticsearch index, which contains documents with their parsed text ([PDF Miner](https://github.com/euske/pdfminer) was used to extract text from the PDF documents). The dataset also contains some, but not all, source PDFs. There are `1109` JSON documents but only `604` PDFs. The PDFs could be used with an altetrnative means of text extraction if desired and new machine-readable data generated for use in modeling. 


