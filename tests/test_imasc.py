import importlib.util


def get_articles(
    jsonl_source="data/microwave_limb_sounder/mls_pubs.jsonl",
    jsonl_output="data/microwave_limb_sounder/article_text.jsonl",
):
    spec = importlib.util.spec_from_file_location(
        "parser.py", "/mnt/c/Users/pavle/Documents/Github/medvidov/IMaSC/parser.py"
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    foo.get_article_text()

    input = jsonl_source
    countin = 0
    with open(input, "r") as f:
        for line in f:
            countin += 1

    output = jsonl_output
    countout = 0
    with open(output, "r") as f:
        for line in f:
            countout += 1

    return countin == countout


def test_get_articles():
    assert get_articles() == True

# Can test to make sure that random article text is the same between both jsonl files
