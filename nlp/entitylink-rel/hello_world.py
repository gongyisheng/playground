from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

wiki_version = "wiki_2019"
base_url = "/media/hdddisk/entitylink-rel-data"

input_text = {
    "my_doc": ("Hello, world!", []),
}

mention_detection = MentionDetection(base_url, wiki_version)
tagger_ner = load_flair_ner("ner-fast")

tagger_ngram = Cmns(base_url, wiki_version, n=5)
mentions, n_mentions = mention_detection.find_mentions(input_text, tagger_ngram)

config = {
    "mode": "eval",
    "model_path": "ed-wiki-2014",
}
model = EntityDisambiguation(base_url, wiki_version, config)

predictions, timing = model.predict(mentions)
result = process_results(mentions, predictions, input_text)
print(result)