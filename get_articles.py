# Script to get article text in JSON form for reference

import json
import random
import os
import codecs
import numpy as np
import re
from re import search

def save_articles_to_file(
    jsonl_source="data/microwave_limb_sounder/mls_pubs.jsonl",
):

    articles = {}
    title = ""
    text = ""

    source = open(jsonl_source)
    i = 0

    for line in source:
        i = i + 1
        j = json.loads(line)
        title = j.get("_source").get("title")
        text = j.get("_source").get("text")

        title = title.replace("/", "")

        filename = "data/microwave_limb_sounder/raw_text/%s%d.txt" % (title, i)

        output = open(filename, "w+")

        output.write(text)
        output.close()

    source.close()
    print(i)

if __name__ == "__main__":
    save_articles_to_file()
