from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_scifact_test():
    print("Retrieving scifact dataset")
    dataset = "scifact"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")
    return GenericDataLoader(data_folder=data_path).load(split="test")

def corpus_to_list(corpus: dict):
    """Takes a dictionary with format [key][text | title] and outputs a list where each entry is the concatenated text and title of a corresponding value."""
    ids = list(corpus.keys())
    text = [
        f"{corpus[cor_id]['title'].strip()} {corpus[cor_id]['text'].strip()}".strip()
        for cor_id in ids
    ]
    return text, ids