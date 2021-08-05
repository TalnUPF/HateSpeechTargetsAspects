import json
from collections import defaultdict
import os
import pandas as pd
import re


def list_reference_concepts(infolder, out_topfolder, path_to_data, path_to_dev_ids, path_to_test_ids):
    main_df = pd.read_csv(path_to_data)

    dev_set_ids = json.load(open(path_to_dev_ids, "r"))
    test_set_ids = json.load(open(path_to_test_ids, "r"))

    excluded_verbs = ["'m", "is", "'re", "am", "are", "be", "'ll", "will", "shall", "were", "was", "been", "'s", "im",
                      "ur", "u", "lol", "sorry", "notsexist", "fuck", "shit", "pls", "please", "t", "s", "'ve", "have",
                      "has"]

    if not os.path.exists(out_topfolder):
        os.makedirs(out_topfolder)
    for inputfile in os.listdir(infolder):
        collection_name = re.sub("_concepts_extracted.json", "", inputfile)
        curclass = collection_name.split("_")[-1]
        curdf = main_df[main_df["Class"] == curclass].copy()
        curdf["text_id"] = range(curdf.shape[0])
        train_set_text_ids = set(
            curdf.loc[sorted(list((set(curdf.index.tolist()) - set(test_set_ids)) - set(dev_set_ids)))][
                "text_id"].tolist())

        concepts = json.load(open(os.path.join(infolder, inputfile), "r"))
        concdict = {}
        for type in concepts:
            concdict[type] = defaultdict(lambda: 0)
            for conc in concepts[type]:
                if (conc['text_id'] in train_set_text_ids and (conc['next_tag'][0] == 'V' or conc['next_tag'] == 'MD')
                        and not conc['next_word'].lower() in excluded_verbs):
                    concdict[type][tuple([conc["text"].lower(), tuple([pos[:2] for pos in conc["postags"].split(" ")])])] += 1
            concdict[type] = dict(concdict[type])

        outfolder = os.path.join(out_topfolder, "concepts_by_pos_%s" % (collection_name))
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        for dictkey in concdict:
            output_filename = outfolder + "/%s_concepts_tf_%s.csv" % (
                dictkey.lower() if dictkey.lower() != "concepts" else "noun_phrase_and_numeric", collection_name)
            df = pd.DataFrame()
            concepts, tf = zip(*list(concdict[dictkey].items()))
            conc, pos = zip(*concepts)
            df["concept"] = list(conc)
            df["pos"] = [" ".join(list(p)) for p in pos]
            df["tf_collection"] = list(tf)
            df = df.sort_values("tf_collection", ascending=False)
            df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    list_reference_concepts(infolder="processed_data/concepts_extracted_with_next_tag_next_word",
                            out_topfolder="processed_data/reference_concepts_followed_by_verb_not_aux",
                            path_to_data="data/all_data_waseem.csv",
                            path_to_dev_ids="data/post_ids_dev.json",
                            path_to_test_ids="data/post_ids_test.json")
