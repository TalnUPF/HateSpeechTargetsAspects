import json
import pandas as pd
import os

from target_aspect_extractor import TargetAspectExtractor, evaluate


def extract_targets_and_aspects_from_waseem():
    dev_set_ids = json.load(open("data/post_ids_dev.json", "r"))
    test_set_ids = json.load(open("data/post_ids_test.json", "r"))
    main_df = pd.read_csv("data/all_data_waseem.csv")

    for curclass in ["racism", "sexism"]:
        curdf = main_df[main_df["Class"] == curclass].copy()
        curdf["text_id"] = range(curdf.shape[0])
        predict_for_ids = set()
        predict_for_ids.update(dev_set_ids)
        predict_for_ids.update(test_set_ids)
        predict_set_text_ids = set(
            curdf.loc[sorted(list(predict_for_ids.intersection(set(curdf.index.tolist()))))]["text_id"].tolist())

        with TargetAspectExtractor() as extractor:
            extractor.get_targets_and_aspects(collection_name="waseem_%s" % (curclass),
                                              collection_name_test="waseem_%s" % (curclass), num_targets_to_take=1,
                                              num_aspects_to_take=1, text_ids_for_prediction=predict_set_text_ids)


def evaluate_targets_and_aspects_waseem(ground_truth_annotation_path, path_to_ids):
    main_df = pd.read_csv("data/all_data_waseem.csv")
    waseem_racism_sexism_df = pd.DataFrame()
    for curclass in ["racism", "sexism"]:
        curdf = main_df[main_df["Class"] == curclass].copy()
        curdf["text_id"] = range(curdf.shape[0])
        sentences_with_offsets = json.load(
            open(os.path.join("processed_data", "input_for_concept_extraction", "waseem_%s.json" % (curclass)), "r"))
        processed_text_ids = sorted(list(set([sent['text_id'] for sent in sentences_with_offsets['sentence_list']])))

        text_id_to_result_dict = {}
        with open(os.path.join("processed_data", "target_aspect_pairs",
                               "target_aspect_pairs_waseem_%s.json" % (curclass)), "r") as fin:
            for dline, line in enumerate(fin):
                curlist = [[pair[0], pair[1]] for pair in json.loads(line)]
                text_id_to_result_dict[processed_text_ids[dline]] = curlist
        curdf["pairs"] = curdf["text_id"].apply(
            lambda x: text_id_to_result_dict[x] if x in text_id_to_result_dict else [])
        waseem_racism_sexism_df = waseem_racism_sexism_df.append(curdf.copy())

    post_ids = json.load(open(path_to_ids, "r"))
    evaluate(ground_truth_annotation_path, waseem_racism_sexism_df.loc[post_ids]["pairs"].tolist())


if __name__ == "__main__":
    extract_targets_and_aspects_from_waseem()
    print("Evaluation dev set.")
    evaluate_targets_and_aspects_waseem("data/annotation_dev.tsv", "data/post_ids_dev.json")
    print("Evaluation test set.")
    evaluate_targets_and_aspects_waseem("data/annotation_test.tsv", "data/post_ids_test.json")
