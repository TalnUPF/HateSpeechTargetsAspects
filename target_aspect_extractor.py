import os
import sys
import json
import pickle
import re
import math
import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict
import requests
from rouge import Rouge


class TargetAspectExtractor:
    """
    Extracts targets and aspects from a collection of posts.
    """

    def __init__(self, ):
        self.target_freq_threshold = 3
        self.valid_types = ["VERB", "ADJ", "concepts"]

        self.use_subj = True
        self.use_tf_idf_for_concepts = True
        self.use_tf_idf_for_appg = True
        self.use_reference_set = True
        self.without_any_subwords = False
        self.without_nominal_subwords = False
        self.use_tf_values = True
        self.beta1 = True
        self.beta2 = True
        self.alpha1 = 1000000  # not from APPG but from "concepts"
        self.alpha2 = 1000
        self.alpha3 = 0.001
        self.alpha4 = 0.000001
        self.alpha5 = False

        self.excluded_aspects = ["'m", "'re", "am", "are", "be", "'ll", "will", "shall", "were", "was", "been", "'s",
                                 "im", "ur", "u", "lol", "sorry", "notsexist", "fuck", "shit", "pls", "please", "t",
                                 "s", "'ve"]
        self.excluded_aspects.extend(
            ["all", "another", "any", "anybody", "anyone", "anything", "as", "aught", "both", "each", "each other",
             "either", "enough", "everybody", "everyone", "everything", "few", "he", "her", "hers", "herself", "him",
             "himself", "his", "i", "idem", "it", "its", "itself", "many", "me", "mine", "most", "my", "myself",
             "naught", "neither", "no one", "nobody", "none", "nothing", "nought", "one", "one another", "other",
             "others", "ought", "our", "ours", "ourself", "ourselves", "several", "she", "some", "somebody", "someone",
             "something", "somewhat", "such", "suchlike", "that", "thee", "their", "theirs", "theirself", "theirselves",
             "them", "themself", "themselves", "there", "these", "they", "thine", "this", "those", "thou", "thy",
             "thyself", "us", "we", "what", "whatever", "whatnot", "whatsoever", "whence", "where", "whereby",
             "wherefrom", "wherein", "whereinto", "whereof", "whereon", "wherever", "wheresoever", "whereto",
             "whereunto", "wherewith", "wherewithal", "whether", "which", "whichever", "whichsoever", "who", "whoever",
             "whom", "whomever", "whomso", "whomsoever", "whose", "whosever", "whosesoever", "whoso", "whosoever", "ye",
             "yon", "yonder", "you", "your", "yours", "yourself", "yourselves"])
        self.excluded_aspects.extend(["gonna", "wanna", "gotta", "shoulda"])

        self.processed_data_folder = "processed_data"
        if not os.path.exists(self.processed_data_folder):
            os.makedirs(self.processed_data_folder)

        self.targets_folder = os.path.join(self.processed_data_folder, "reference_target_candidates")
        if not os.path.exists(self.targets_folder):
            os.makedirs(self.targets_folder)

        self.target_aspect_pairs_output = os.path.join(self.processed_data_folder, "target_aspect_pairs")
        if not os.path.exists(self.target_aspect_pairs_output):
            os.makedirs(self.target_aspect_pairs_output)

        self.Ndoc = 161105350

    def __enter__(self, ):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def load_parsed_sentences(self, predict_set_text_ids=[]):
        sentences_with_offsets = json.load(open(os.path.join(self.processed_data_folder, "input_for_concept_extraction",
                                                             "%s.json" % (self.collection_name_test)), "r"))
        curtext = 0
        texts = [[]]
        take_text = []
        for sent_id_int, sent in enumerate(sentences_with_offsets["sentence_list"]):
            sent_id = str(sent_id_int)
            if int(sent["text_id"]) != curtext:
                take_text.append(True if curtext in predict_set_text_ids or len(predict_set_text_ids) == 0 else False)
                curtext = int(sent["text_id"])
                texts.append([])
            texts[-1].append(sent_id)
        take_text.append(True if curtext in predict_set_text_ids or len(predict_set_text_ids) == 0 else False)
        return sentences_with_offsets, texts, take_text

    def get_reference_target_candidates(self, ):
        targets_ce_inputfile = ""
        if self.use_subj:
            targets_ce_inputfile = os.path.join(self.processed_data_folder,
                                                "reference_concepts_followed_by_verb_not_aux",
                                                "concepts_by_pos_%s" % (self.collection_name),
                                                "noun_phrase_and_numeric_concepts_tf_%s.csv" % (self.collection_name))
        else:
            targets_ce_inputfile = os.path.join(self.processed_data_folder, "reference_concepts_followed_by_any_pos",
                                                "concepts_by_pos_%s" % (self.collection_name),
                                                "noun_phrase_and_numeric_concepts_tf_%s.csv" % (self.collection_name))
        if not os.path.exists(targets_ce_inputfile):
            print("Extract concepts from the reference set with https://github.com/TalnUPF/ConceptExtraction/ first.")
            assert os.path.exists(targets_ce_inputfile)
        df = pd.read_csv(targets_ce_inputfile)
        target_candidates_df = df[df["tf_collection"] >= self.target_freq_threshold]
        reference_target_candidates = target_candidates_df[target_candidates_df["pos"] != "CD"]["concept"].tolist()
        with open(os.path.join(self.targets_folder, "targets_%s.csv" % (self.collection_name)), "wb") as fout:
            fout.write(("\n".join(reference_target_candidates) + "\n").encode("utf8"))
        target_rank_dict = {}
        next_rank = 0
        for conc in reference_target_candidates:
            if not conc in target_rank_dict:
                target_rank_dict[conc] = next_rank
                next_rank += 1
        return target_rank_dict

    def get_sents_target_aspect_candidates_dict(self, concepts):
        sents_target_aspect_candidates_dict = {}
        for type in self.valid_types:
            for conc in concepts[type]:
                if not conc["text_id"] in sents_target_aspect_candidates_dict:
                    sents_target_aspect_candidates_dict[conc["text_id"]] = defaultdict(list)
                concept_field = "concept" if "concept" in conc else "text"
                if not conc[concept_field].lower() in self.excluded_aspects:
                    sents_target_aspect_candidates_dict[conc["text_id"]]["%s_aspects" % (type)].append(tuple(
                        [conc[concept_field].lower(), conc["begin"], conc["end"], conc[concept_field],
                         conc["postags"]]))
        return sents_target_aspect_candidates_dict

    def get_invert_dict(self, sents_target_aspect_candidates_dict):
        invert_dict = {}
        target_aspect_counter = {}
        aspect_set = {}
        for type in self.valid_types:
            target_aspect_counter[type] = defaultdict(lambda: 0)
            aspect_set[type] = set()
        for text_id in sents_target_aspect_candidates_dict:
            invert_dict[text_id] = {}
            for type in self.valid_types:
                invert_dict[text_id][type] = {}
            for type in self.valid_types:
                for aspect in set(sents_target_aspect_candidates_dict[text_id]["%s_aspects" % (type)]):
                    invert_dict[text_id][type][aspect[1]] = aspect
        return invert_dict

    def get_tfidf_ranks(self, ):
        tfidf = {}
        maxtfidf = {}
        tfidf_path = os.path.join(self.processed_data_folder,
                                  "tfidf_subtr_from_maxtfidf_%s.pickle" % (self.collection_name))
        if os.path.exists(tfidf_path):
            tfidf, maxtfidf = pickle.load(open(tfidf_path, "rb"))
        else:
            for type in self.valid_types:
                gigaword_stats_filepath = os.path.join(self.processed_data_folder, "gigaword_stats",
                                                       "concepts_by_pos_%s" % (self.collection_name),
                                                       "%s_concepts_tf_%s_tf_idf.csv" % (
                                                       type.lower() if type != "concepts" else "noun_phrase_and_numeric",
                                                       self.collection_name))
                if not os.path.exists(gigaword_stats_filepath):
                    print("Run count_gigaword_stats_solr.py first.")
                    assert os.path.exists(gigaword_stats_filepath)
                tfidf_df = pd.read_csv(gigaword_stats_filepath)
                maxtfidf[type] = tfidf_df["TF-IDF"].tolist()[0] + 1
                tfidf_df["TF-IDF"] = maxtfidf[type] - tfidf_df["TF-IDF"]
                tfidf[type] = dict(zip(tfidf_df["concept"].tolist(), tfidf_df["TF-IDF"].tolist()))
        return tfidf, maxtfidf

    def get_targets_and_aspects_candidates_extended(self, invert_dict, sentences_with_offsets, sent_ids, tfidf,
                                                    maxtfidf, next_sent):
        aspects_tfidf = {}
        for type in self.valid_types:
            aspects_tfidf[type] = []
        skip_until = 0
        for sent_id in sent_ids:
            next_sent += 1
            sent = sentences_with_offsets["sentence_list"][next_sent]
            text_id = int(sent["text_id"])
            for token in sent["token_list"]:
                if token["beg_offset"] < skip_until:
                    continue
                is_candidate = False
                if text_id in invert_dict:
                    for type in self.valid_types:
                        if token["beg_offset"] in invert_dict[text_id][type] and \
                                invert_dict[text_id][type][token["beg_offset"]][4] != "CD":
                            aspect = invert_dict[text_id][type][token["beg_offset"]]
                            if not aspect[0] in self.excluded_aspects and len(aspect[0]) > 1:
                                if not aspect[0] in tfidf[type]:
                                    res = requests.get(
                                        "http://clasificador-taln.upf.edu/index/english_giga/select?q=text:\"%20" +
                                        aspect[0].replace("%", "%25").replace(" ", "%20") + "%20\"").json()
                                    aspect_df = 0
                                    if "response" in res and "numFound" in res["response"]:
                                        aspect_df = res["response"]["numFound"]
                                    curtfidf = 1.0 * math.log(1.0 + 1.0) * (
                                                math.log(1.0 * self.Ndoc / (1.0 + aspect_df)) + 1)
                                    tfidf[type][aspect[0]] = maxtfidf[type] - curtfidf
                                    tfidf[type][aspect[0]] = tfidf[type][aspect[0]] if tfidf[type][aspect[0]] > 0 else 1.0
                                skip_until = aspect[2]
                                is_candidate = True
                                aspects_tfidf[type].append(tuple([aspect[0], aspect[1],
                                                                  tfidf[type][aspect[0]] if not aspect[0] in ["nan", "null"] else 999999999,
                                                                  aspect[3], aspect[4]]))
                                break
                if not is_candidate:
                    skip_until = token["end_offset"]
        return aspects_tfidf, tfidf

    def extract_targets_and_aspects(self, reference_target_candidates_ranked, sentences_with_offsets, sent_ids,
                                    next_sent, aspects_tfidf, num_targets_to_take=3, num_aspects_to_take=3):
        target_core_ids_used = set()
        aspect_core_ids_used = set()
        target_core_ids_used_local = set()
        target_aspect_pairs = []
        for repeat in range(num_targets_to_take):
            aspect_core_ids_used = set()
            target_core_ids_used.update(target_core_ids_used_local)
            target_core_ids_used_local = set()
            for repeat2 in range(num_aspects_to_take):
                target_core = ""
                target_core_id = -1
                aspect_core = ""
                aspect_core_id = -1
                best_target = tuple(["none", -1, 999999999, "none", "none"])
                best_aspect = tuple(["none", -1, 999999999, "none", "none"])

                best_target_from_targets_list = False
                aspects_with_target_words = []
                if self.use_reference_set:
                    for aspect_tfidf in list(set(chain.from_iterable([aspects_tfidf["concepts"]]))):
                        for word in [word for word in re.sub("_", " ", aspect_tfidf[0]).split(" ") if len(word) > 2]:
                            for target_key in reference_target_candidates_ranked:
                                if target_key != word and (self.without_nominal_subwords or self.without_any_subwords):
                                    continue
                                if target_key.startswith(word) or word.startswith(target_key):
                                    aspects_with_target_words.append(tuple([aspect_tfidf[0], aspect_tfidf[1],
                                        (reference_target_candidates_ranked[target_key] if self.use_tf_values else 0)
                                        - self.alpha1
                                        - (self.alpha3 if target_key.lower() == word.lower() else 0)
                                        - (self.alpha2 if aspect_tfidf[3] !=aspect_tfidf[0] else 0)
                                        + (self.alpha4 * aspect_tfidf[1]), aspect_tfidf[3],
                                        int(word in aspect_tfidf[0].split(" ")[0].split("_"))]))
                                else:
                                    for target_key_word in [target_key_word for target_key_word in target_key.split(" ")
                                                            if len(target_key_word) > 2]:
                                        if ((not ('\\' in word)) and
                                                (re.search("^(%s)" % re.sub("\*", "", word[:5]), target_key_word)
                                                 or re.search("^(%s)" % re.sub("\*", "", target_key_word[:5]), word))):
                                            aspects_with_target_words.append(tuple([aspect_tfidf[0], aspect_tfidf[1],
                                                (reference_target_candidates_ranked[target_key] if self.use_tf_values else 0)
                                                - self.alpha1
                                                - (self.alpha3 if target_key.lower() == word.lower() else 0)
                                                - (self.alpha2 if aspect_tfidf[3] != aspect_tfidf[0] else 0)
                                                + (self.alpha4 * aspect_tfidf[1]),
                                                aspect_tfidf[3], int(word in aspect_tfidf[0].split(" ")[0].split("_"))]))
                                            break
                    for aspect_tfidf in list(set(chain.from_iterable(
                            [aspects_tfidf["ADJ"], [asp for asp in aspects_tfidf["VERB"] if asp[4] in ["VBN"]]]))):
                        for word in [word for word in re.sub("_", " ", aspect_tfidf[0]).split(" ") if len(word) > 2]:
                            for target_key in reference_target_candidates_ranked:
                                if target_key != word and self.without_any_subwords:
                                    continue
                                if target_key.startswith(word) or word.startswith(target_key):
                                    aspects_with_target_words.append(tuple([aspect_tfidf[0], aspect_tfidf[1],
                                        (reference_target_candidates_ranked[target_key] if self.use_tf_values else 0)
                                        - (self.alpha3 if target_key.lower() == word.lower() else 0)
                                        - (self.alpha2 if aspect_tfidf[3] != aspect_tfidf[0] else 0)
                                        + (self.alpha4 * aspect_tfidf[1]),
                                        aspect_tfidf[3], int(word in aspect_tfidf[0].split(" ")[0].split("_"))]))
                                else:
                                    for target_key_word in [target_key_word for target_key_word in target_key.split(" ")
                                                            if len(target_key_word) > 2]:
                                        if ((not ('\\' in word)) and
                                            (re.search("^(%s)" % re.sub("\*", "", word[:5]), target_key_word) or
                                             re.search("^(%s)" % re.sub("\*", "", target_key_word[:5]), word))):
                                            aspects_with_target_words.append(tuple([aspect_tfidf[0], aspect_tfidf[1],
                                                (reference_target_candidates_ranked[target_key] if self.use_tf_values else 0)
                                                - (self.alpha3 if target_key.lower() == word.lower() else 0)
                                                - (self.alpha2 if aspect_tfidf[3] != aspect_tfidf[0] else 0)
                                                + (self.alpha4 * aspect_tfidf[1]),
                                                aspect_tfidf[3], int(word in aspect_tfidf[0].split(" ")[0].split("_"))]))
                                            break
                best_target_first_pos = ""
                if len(aspects_with_target_words) > 0:
                    best_target_list = sorted(aspects_with_target_words, key=lambda x: x[2])
                    btl_i = 0
                    while btl_i < len(best_target_list) and best_target_list[btl_i][1] in target_core_ids_used:
                        btl_i += 1
                    if btl_i < len(best_target_list):
                        best_target = list(best_target_list[btl_i])[:]
                        target_core = best_target[0]
                        target_core_id = best_target[1]
                        best_target_first_pos = ""
                        best_target_from_targets_list = True
                for i in range(2 if best_target[0] == "none" else 1):
                    if best_target[0] != "none" or best_target_from_targets_list:
                        best_target = list(best_target)[:]
                        found = False
                        for sent_id in range(next_sent - len(sent_ids) + 1, next_sent + 1):
                            sent = sentences_with_offsets["sentence_list"][sent_id]
                            for dtok, token in enumerate(sent["token_list"]):
                                if token["beg_offset"] == best_target[1]:
                                    best_target_first_pos = token["tag"]
                                if (isinstance(best_target[1], int)
                                        and token["beg_offset"] >= best_target[1] + len(best_target[3])):
                                    found = True
                                    break
                            if found:
                                break
                        if found and self.beta1:
                            can_adj = False
                            can_verb = True
                            for token in sent["token_list"][dtok:]:
                                if token['token'].lower() in ['in', 'at', 'on', 'of', 'for', ',', 'and', "'s", "'", "s",
                                                              "with"] or token['tag'] == "DT":
                                    can_adj = True
                                    can_verb = True
                                    best_target[0] += " " + token['token']
                                elif can_adj and token['tag'][0] == 'J':
                                    best_target[0] += " " + token['token']
                                    can_verb = False
                                elif token['tag'][0] == 'N':
                                    can_adj = False
                                    best_target[0] += " " + token['token']
                                elif can_verb and token['tag'][0] == 'V' and token['token'].lower().endswith("ing"):
                                    can_adj = True
                                    best_target[0] += " " + token['token']
                                else:
                                    break

                    aspects_tfidf_concepts = []
                    window_size = 15
                    if best_target_from_targets_list or best_target[0] != "none":
                        aspects_tfidf_concepts = [aspect_tfidf for aspect_tfidf in
                                                  list(set(chain.from_iterable([aspects_tfidf["concepts"]]))) if (
                                                          (aspect_tfidf[1] < best_target[1] and aspect_tfidf[1] > best_target[1] - window_size)
                                                          or (aspect_tfidf[1] > best_target[1] + len(best_target[0])
                                                              and aspect_tfidf[1] < best_target[1] + len(best_target[0]) + 2000))]
                        aspects_before_target = [aspect_tfidf for aspect_tfidf in list(set(chain.from_iterable([
                            [asp for asp in aspects_tfidf["VERB"] if (asp[1] < best_target[1]
                             and asp[1] > best_target[1] - window_size and asp[4] in ["VBN"])]])))
                                    if ((aspect_tfidf[1] < best_target[1] and aspect_tfidf[1] > best_target[1] - window_size)
                                        or (aspect_tfidf[1] > best_target[1] + len(best_target[0])
                                            and aspect_tfidf[1] < best_target[1] + len(best_target[0]) + 2000))]
                        if len(aspects_before_target) > 0:
                            aspects_tfidf_concepts = aspects_before_target
                            found = True
                        else:
                            aspects_after_target = [aspect_tfidf for aspect_tfidf in aspects_tfidf["concepts"] if
                                                    aspect_tfidf[1] > best_target[1] + len(best_target[0]) and
                                                    aspect_tfidf[4][0] == 'N']
                            if len(aspects_after_target) > 0:
                                aspects_tfidf_concepts = aspects_after_target
                                found = True
                            else:
                                aspects_after_target = [aspect_tfidf for aspect_tfidf in
                                                        list(set(chain.from_iterable([aspects_tfidf["ADJ"]]))) if ((
                                                                (aspect_tfidf[1] < best_target[1] and aspect_tfidf[1] > best_target[1] - window_size)
                                                                or (aspect_tfidf[1] > best_target[1] + len(best_target[0])
                                                                    and aspect_tfidf[1] < best_target[1] + len(best_target[0]) + 2000)) and
                                                        aspect_tfidf[4] == 'JJ')]
                                if len(aspects_after_target) > 0:
                                    aspects_tfidf_concepts = aspects_after_target
                                    found = True
                        if not found:
                            aspects_tfidf_concepts = [aspect_tfidf for aspect_tfidf in
                                                      list(set(chain.from_iterable([aspects_tfidf["concepts"]]))) if
                                                      aspect_tfidf[1] < best_target[1] or aspect_tfidf[1] > best_target[1] + len(best_target[0])]
                    if len(aspects_tfidf["concepts"]) > 0 or len(aspects_tfidf_concepts) > 0:
                        if i > 0 or best_target[0] != "none":
                            best_aspect_list = []
                            if len(aspects_tfidf_concepts) > 0:
                                best_aspect_list = [conc for conc in sorted(aspects_tfidf_concepts,
                                                                            key=lambda x: abs(x[1] - best_target[1]))]
                            else:
                                best_aspect_list = [conc for conc in sorted(aspects_tfidf["concepts"],
                                                                            key=lambda x: abs(x[1] - best_target[1]))]
                            if len(best_aspect_list) != 0:
                                btl_i = 0
                                while btl_i < len(best_aspect_list) and best_aspect_list[btl_i][1] in aspect_core_ids_used:
                                    btl_i += 1
                                if btl_i < len(best_aspect_list):
                                    best_aspect = best_aspect_list[btl_i]
                                    aspect_core = best_aspect[0]
                                    aspect_core_id = best_aspect[1]
                            else:
                                best_aspect = best_target[:]
                                aspect_core = target_core
                                aspect_core_id = best_aspect[1]
                        if i == 0 and best_target[0] == "none":
                            best_aspect = tuple(["none", -1, 999999999, "none", "none"])
                            best_target = tuple(["none", -1, 999999999, "none", "none"])
                            if self.alpha1 > 0:
                                select_from_list = aspects_tfidf["concepts"][:]
                            else:
                                select_from_list = list(chain.from_iterable(
                                    [aspects_tfidf["concepts"], aspects_tfidf["ADJ"], aspects_tfidf["VERB"]]))
                            if len(select_from_list) > 0:
                                if self.use_tf_idf_for_concepts:
                                    best_target_list = []
                                    asp_list = [asp for asp in select_from_list if len(asp[0]) > 2]
                                    if len(asp_list) == 0:
                                        asp_list = [asp for asp in select_from_list if len(asp[0]) > 1]
                                        if len(asp_list) == 0:
                                            asp_list = [asp for asp in select_from_list if len(asp[0]) > 0]

                                    if any(([asp for asp in asp_list if asp[3] != asp[0]])):
                                        best_target_list = sorted([asp for asp in asp_list if asp[3] != asp[0]],
                                                                  key=lambda x: x[2])
                                    else:
                                        best_target_list = sorted(asp_list, key=lambda x: x[2])
                                    btl_i = 0
                                    while btl_i < len(best_target_list) and best_target_list[btl_i][1] in target_core_ids_used:
                                        btl_i += 1
                                    if btl_i < len(best_target_list):
                                        best_target = list(best_target_list[btl_i])[:]
                                        target_core = best_target[0]
                                        target_core_id = best_target[1]
                                else:
                                    np.random.shuffle(select_from_list)
                                    best_target_list = [asp for asp in select_from_list if len(asp[0]) > 2]
                                    if len(best_target_list) == 0:
                                        best_target_list = [asp for asp in select_from_list if len(asp[0]) > 1]
                                        if len(best_target_list) == 0:
                                            best_target_list = select_from_list
                                    btl_i = 0
                                    while btl_i < len(best_target_list) and best_target_list[btl_i][1] in target_core_ids_used:
                                        btl_i += 1
                                    if btl_i < len(best_target_list):
                                        best_target = list(best_target_list[btl_i])[:]
                                        target_core = best_target[0]
                                        target_core_id = best_target[1]
                            best_aspect = tuple(["none", -1, 999999999, "none", "none"])
                    elif len(aspects_tfidf["ADJ"]) > 0 and self.use_tf_idf_for_appg:
                        best_aspect = tuple(["none", -1, 999999999, "none", "none"])
                        best_aspect_list = []
                        if self.use_tf_idf_for_appg:
                            best_aspect_list = sorted(aspects_tfidf["ADJ"], key=lambda x: x[2])
                        else:
                            select_from_list = aspects_tfidf["ADJ"][:]
                            np.random.shuffle(select_from_list)
                            best_aspect_list = select_from_list
                        if best_target[0] == "none":
                            btl_i = 0
                            while btl_i < len(best_aspect_list) and best_aspect_list[btl_i][1] in target_core_ids_used:
                                btl_i += 1
                            if btl_i < len(best_aspect_list):
                                best_target = list(best_aspect_list[btl_i])[:]
                                target_core = best_target[0]
                                target_core_id = best_target[1]
                        else:
                            btl_i = 0
                            while btl_i < len(best_aspect_list) and best_aspect_list[btl_i][1] in aspect_core_ids_used:
                                btl_i += 1
                            if btl_i < len(best_aspect_list):
                                best_aspect = list(best_aspect_list[btl_i])[:]
                                aspect_core = best_aspect[0]
                                aspect_core_id = best_aspect[1]
                    elif len(aspects_tfidf["VERB"]) > 0 and self.use_tf_idf_for_appg:
                        best_aspect = []
                        best_aspect_list = []
                        if self.use_tf_idf_for_appg:
                            best_aspect_list = sorted(aspects_tfidf["VERB"], key=lambda x: x[2])
                        else:
                            select_from_list = aspects_tfidf["VERB"][:]
                            np.random.shuffle(select_from_list)
                            best_aspect_list = select_from_list
                        if best_target[0] == "none":
                            btl_i = 0
                            while btl_i < len(best_aspect_list) and best_aspect_list[btl_i][1] in target_core_ids_used:
                                btl_i += 1
                            if btl_i < len(best_aspect_list):
                                best_target = list(best_aspect_list[btl_i])[:]
                                target_core = best_target[0]
                                target_core_id = best_target[1]
                        else:
                            btl_i = 0
                            while btl_i < len(best_aspect_list) and best_aspect_list[btl_i][1] in aspect_core_ids_used:
                                btl_i += 1
                            if btl_i < len(best_aspect_list):
                                best_aspect = list(best_aspect_list[btl_i])[:]
                                aspect_core = best_aspect[0]
                                aspect_core_id = best_aspect[1]
                    if (not best_target) or best_target[0] == "none" or (not best_aspect) or best_aspect[0] == "none":
                        if not best_target or best_target[0] == "none":
                            best_target = tuple(["none", -1, 999999999, "none", "none"])
                            best_target_list_c = sorted([asp for asp in aspects_tfidf["concepts"] if len(asp[0]) > 2],
                                                        key=lambda x: x[2], reverse=True)
                            best_target_list_a = sorted(aspects_tfidf["ADJ"], key=lambda x: x[2], reverse=True)
                            best_target_list_v = sorted(aspects_tfidf["VERB"], key=lambda x: x[2])
                            best_target_list_t = [
                                [curtoken["token"].lower(), curtoken["beg_offset"], -1, curtoken["token"], True]
                                for curtoken in
                                sentences_with_offsets["sentence_list"][next_sent - len(sent_ids) + 1]["token_list"]]
                            best_target_list = list(chain.from_iterable(
                                [best_target_list_c, best_target_list_a, best_target_list_v, best_target_list_t]))
                            btl_i = 0
                            while btl_i < len(best_target_list) and best_target_list[btl_i][1] in target_core_ids_used:
                                btl_i += 1
                            if btl_i < len(best_target_list):
                                best_target = list(best_target_list[btl_i])[:]
                                target_core = best_target[0]
                                target_core_id = best_target[1]
                            if best_target[0] == "none":
                                best_target = list(best_target_list[0])[:]
                                target_core = best_target[0]
                                target_core_id = best_target[1]
                        else:
                            best_aspect = list(best_target)[:]
                            aspect_core = target_core
                            aspect_core_id = best_target[1]

                # Aspect forward prolongation
                best_aspect = list(best_aspect)
                found = False
                found_sent = False
                dtok_begin = -1
                for sent_id in range(next_sent - len(sent_ids) + 1, next_sent + 1):
                    sent = sentences_with_offsets["sentence_list"][sent_id]
                    for dtok, token in enumerate(sent["token_list"]):
                        if token["beg_offset"] == best_aspect[1]:
                            dtok_begin = dtok
                            found_sent = True
                        if token["beg_offset"] >= best_aspect[1] + len(best_aspect[3]):
                            found = True
                            break
                    if found or found_sent:
                        break
                if found or found_sent:
                    best_aspect[0] = sent["token_list"][dtok_begin]['token'].lower()
                    can_adj = False
                    can_verb = True
                    for token in sent["token_list"][dtok_begin + 1:]:
                        if token['token'].lower() in ['in', 'at', 'on', 'of', 'for', ',', 'and', "'s", "'", "s", "that",
                                                      "is", "are", "with"] or token['tag'] == "DT":
                            can_adj = True
                            can_verb = True
                            best_aspect[0] += " " + token['token']
                        elif can_adj and token['tag'][0] == 'J':
                            best_aspect[0] += " " + token['token']
                            can_verb = False
                        elif token['tag'][0] == 'N':
                            can_adj = False
                            best_aspect[0] += " " + token['token']
                        elif can_verb and token['tag'][0] == 'V' and token['token'].lower().endswith("ing"):
                            can_adj = True
                            best_aspect[0] += " " + token['token']
                        else:
                            break
                best_aspect = list(best_aspect)
                best_target = list(best_target)
                if best_target[0] != "none" and best_aspect[0] != "none":
                    # Aspect backward prolongation
                    found = False
                    dtok_begin_aspect = -1
                    sent = []
                    for sent_id in range(next_sent - len(sent_ids) + 1, next_sent + 1):
                        sent = sentences_with_offsets["sentence_list"][sent_id]
                        for dtok, token in enumerate(sent["token_list"]):
                            if token["beg_offset"] == best_aspect[1]:
                                dtok_begin_aspect = dtok
                                found = True
                                break
                        if found:
                            break

                    dtok_after_target = -1
                    found_t = False
                    for dtok, token in enumerate(sent["token_list"]):
                        if (isinstance(best_target[1], int)
                                and token["beg_offset"] >= best_target[1] + len(best_target[3])):
                            dtok_after_target = dtok
                            found_t = True
                            break
                        if found_t:
                            break
                    if found:
                        can_adj = True
                        can_noun = True
                        for token in reversed(sent["token_list"][(
                        max(0, dtok_after_target) if dtok_after_target < dtok_begin_aspect else 0):dtok_begin_aspect]):
                            if token['token'].lower() in ['in', 'at', 'on', 'of', 'for', ',', 'and', 'to', ":", "/",
                                                          "with", "by"] or token['tag'] in ["DT", "RB", "CD"]:
                                can_adj = True
                                can_noun = True
                                best_aspect[0] = token['token'] + " " + best_aspect[0]
                            elif can_adj and token['tag'][0] == 'J':
                                best_aspect[0] = token['token'] + " " + best_aspect[0]
                                can_noun = False
                            elif token['tag'][0] == 'N':
                                can_adj = True
                                best_aspect[0] = token['token'] + " " + best_aspect[0]
                            elif token['tag'][0] == 'V' and not token['token'].lower() in ['are', 'is', 'am', 'was',
                                                                                           'were', 'been']:
                                can_adj = False
                                best_aspect[0] = token['token'] + " " + best_aspect[0]
                            else:
                                break

                    for dtok, token in enumerate(sent["token_list"]):
                        if token["beg_offset"] == best_target[1]:
                            best_target_first_pos = token["tag"]
                            break
                    if (best_target_first_pos and best_target_first_pos[0] == "J" and " " in best_target[0]
                            and not best_target[4]):
                        best_aspect = list(best_aspect)
                        best_target = list(best_target)
                        if not best_target[1] in aspect_core_ids_used:
                            best_aspect[0] = best_target[0].split(" ")[0]
                            best_target[0] = " ".join(best_target[0].split(" ")[1:])
                            aspect_core = best_aspect[0]
                            aspect_core_id = best_target[1]
                        temp = best_target[0].split(" ")[0]
                        best_target[0] = " ".join(best_target[0].split(" ")[1:])
                        if target_core.startswith(temp):
                            target_core = target_core[len(temp) + 1:]

                    not_end = ['in', 'at', 'on', 'of', 'for', ',', 'and', "'s", "'", "s", "that", "is", "are", "the",
                               "an", "a", "with", "'re", "'m", "'ll", "'ve", "by"]
                    for dword, word in enumerate(reversed(best_target[0].split(" "))):
                        if not word in not_end:
                            break
                    if dword != 0:
                        best_target[0] = " ".join(best_target[0].split(" ")[:-dword])
                    for dword, word in enumerate(reversed(best_aspect[0].split(" "))):
                        if not word in not_end:
                            break
                    if dword != 0:
                        best_aspect[0] = " ".join(best_aspect[0].split(" ")[:-dword])

                    for dword, word in enumerate(best_target[0].split(" ")):
                        if not word in not_end:
                            break
                    if dword != 0:
                        best_target[0] = " ".join(best_target[0].split(" ")[dword:])
                    for dword, word in enumerate(best_aspect[0].split(" ")):
                        if not word in not_end:
                            break
                    if dword != 0:
                        best_aspect[0] = " ".join(best_aspect[0].split(" ")[dword:])

                    if self.alpha5 and self.beta2:
                        target_aspect_pairs.append(tuple([target_core.lower(), aspect_core.lower()]))
                    elif self.alpha5:
                        target_aspect_pairs.append(tuple([best_target[0].lower(), aspect_core.lower()]))
                    elif self.beta2:
                        target_aspect_pairs.append(tuple([target_core.lower(), best_aspect[0].lower()]))
                    else:
                        target_aspect_pairs.append(tuple([best_target[0].lower(), best_aspect[0].lower()]))
                    target_core_ids_used_local.update([target_core_id])
                    aspect_core_ids_used.update([aspect_core_id])
        return target_aspect_pairs

    def get_targets_and_aspects(self, collection_name, collection_name_test, num_targets_to_take=1,
                                num_aspects_to_take=1, text_ids_for_prediction=[]):
        self.collection_name = collection_name
        self.collection_name_test = collection_name_test

        concepts_inputfile = os.path.join(self.processed_data_folder, "concepts_extracted_with_next_tag_next_word",
                                          "%s_concepts_extracted.json" % (collection_name_test))
        if not os.path.exists(concepts_inputfile):
            print("Extract concepts from the test set with https://github.com/TalnUPF/ConceptExtraction/ first.")
            assert os.path.exists(concepts_inputfile)

        concepts = json.load(open(concepts_inputfile, "r"))
        sentences_with_offsets, texts, take_text = self.load_parsed_sentences(text_ids_for_prediction)

        reference_target_candidates_ranked = self.get_reference_target_candidates()

        sents_target_aspect_candidates_dict = self.get_sents_target_aspect_candidates_dict(concepts)
        invert_dict = self.get_invert_dict(sents_target_aspect_candidates_dict)

        tfidf, maxtfidf = self.get_tfidf_ranks()

        target_aspect_pairs = []
        next_sent = -1
        for dtext, sent_ids in enumerate(texts):
            if not take_text[dtext]:
                target_aspect_pairs.append([tuple(["", ""])])
                next_sent += len(sent_ids)
                continue
            print("Post is being analyzed: %d/%d." % (dtext + 1, len(texts)))
            aspects_tfidf, tfidf = self.get_targets_and_aspects_candidates_extended(invert_dict, sentences_with_offsets,
                                                                                    sent_ids, tfidf, maxtfidf,
                                                                                    next_sent)
            next_sent += len(sent_ids)
            pickle.dump([tfidf, maxtfidf], open(
                os.path.join(self.processed_data_folder, "tfidf_subtr_from_maxtfidf_%s.pickle" % (collection_name)), "wb"))
            pickle.dump([tfidf, maxtfidf], open(os.path.join(self.processed_data_folder,
                                                  "tfidf_subtr_from_maxtfidf_%s_backup.pickle" % (collection_name)), "wb"))
            target_aspect_pairs.append(
                self.extract_targets_and_aspects(reference_target_candidates_ranked, sentences_with_offsets, sent_ids,
                                                 next_sent, aspects_tfidf, num_targets_to_take, num_aspects_to_take))

        with open(os.path.join(self.target_aspect_pairs_output, "target_aspect_pairs_%s.json" % (collection_name_test)), "w") as fout:
            for target_aspect_pairs_for_text in target_aspect_pairs:
                json.dump(target_aspect_pairs_for_text, fout)
                fout.write("\n")

    def evaluate_collection(self, ground_truth_annotation_path, collection_name_test):
        predicted_pairs = []
        with open(os.path.join(self.target_aspect_pairs_output, "target_aspect_pairs_%s.json" % (collection_name_test)),
                  "r") as fin:
            for dline, line in enumerate(fin):
                predicted_pairs.append([[pair[0], pair[1]] for pair in json.loads(line)])
        evaluate(ground_truth_annotation_path, predicted_pairs)


def evaluate(ground_truth_annotation_path, predicted_pairs):
    rouge = Rouge()
    eval_data = pd.read_csv(ground_truth_annotation_path, sep="\t", header=0, names=[0, 1, 2, 3, 4, 5])
    eval_data["target"] = eval_data.apply(
        lambda x: (x[0] + (" " + x[2] if x[2] == x[2] else "") + (" " + x[4] if x[4] == x[4] else "")).lower(), axis=1)
    eval_data["target_exists"] = eval_data["target"].apply(lambda x: x.strip() != "-" and x.strip() != "")
    eval_data["aspect"] = eval_data.apply(
        lambda x: (str(x[1]) + (" " + x[3] if x[3] == x[3] else "") + (" " + x[5] if x[5] == x[5] else "")).lower(), axis=1)
    eval_data["aspect_exists"] = eval_data["aspect"].apply(lambda x: x.strip() != "-" and x.strip() != "")
    eval_data["predicted_pairs"] = predicted_pairs
    eval_data["target_partial_match"] = eval_data.apply(lambda x: int(
        any((word.lower() in x["target"].lower() for pair in x["predicted_pairs"] for word in pair[0].split(" ") if
             not word in ['', 'in', 'at', 'on', 'of', 'for', ',', 'and', "'s", "'", "s", "that", "is", "are", "the",
                          "an", "a", "with", "am", "was", "were", "'m", "'ll", "'ve", "by"]))), axis=1)
    eval_data["aspect_partial_match"] = eval_data.apply(lambda x: int(
        any((word.lower() in x["aspect"].lower() for pair in x["predicted_pairs"] for word in pair[1].split(" ") if
             not word in ['', 'in', 'at', 'on', 'of', 'for', ',', 'and', "'s", "'", "s", "that", "is", "are", "the",
                          "an", "a", "with", "am", "was", "were", "'m", "'ll", "'ve", "by"]))), axis=1)
    eval_data["target_exact_match"] = eval_data.apply(lambda x: int(any((x[field] == x[field] and x[field].strip()
            and x[field].strip() != '-' and x["predicted_pairs"][0][0] == re.sub("(#|@)", "", x[field].lower())
            for field in [0, 2, 4]))) if len(x["predicted_pairs"]) > 0 else 0, axis=1)
    eval_data["aspect_exact_match"] = eval_data.apply(lambda x: int(any((x[field] == x[field] and x[field].strip()
            and x[field].strip() != '-' and x["predicted_pairs"][0][1] == re.sub("(#|@)", "", x[field].lower())
            for field in [1, 3, 5]))) if len(x["predicted_pairs"]) > 0 else 0, axis=1)
    eval_data["rouges_targetsLp"] = eval_data.apply(
        lambda x: max([rouge.get_scores(x["predicted_pairs"][iauto][0], re.sub("(#|@)", "", x[field].lower()))[0]['rouge-l']["p"]
                       if len(x["predicted_pairs"][iauto][0]) > 0 else 0
                       for field in [0, 2, 4] if x[field] == x[field] and x[field].strip() and x[field].strip() != '-'
                           for iauto in range(len(x["predicted_pairs"]))]) if len(x["predicted_pairs"]) > 0 and any(
        (x[field] == x[field] and x[field].strip() and x[field].strip() != '-' for field in [0, 2, 4])) else -1, axis=1)
    eval_data["rouges_aspectsLp"] = eval_data.apply(
        lambda x: max([rouge.get_scores(x["predicted_pairs"][iauto][1], re.sub("(#|@)", "", x[field].lower()))[0]['rouge-l']["p"]
                       if len(x["predicted_pairs"][iauto][1]) > 0 else 0
                       for field in [1, 3, 5] if x[field] == x[field] and x[field].strip() and x[field].strip() != '-'
                           for iauto in range(len(x["predicted_pairs"]))]) if len(x["predicted_pairs"]) > 0 and any(
        (x[field] == x[field] and x[field].strip() and x[field].strip() != '-' for field in [1, 3, 5])) else -1, axis=1)
    eval_data["rouges_targetsLr"] = eval_data.apply(
        lambda x: max([rouge.get_scores(x["predicted_pairs"][iauto][0], re.sub("(#|@)", "", x[field].lower()))[0]['rouge-l']["r"]
                       if len(x["predicted_pairs"][iauto][0]) > 0 else 0
                       for field in [0, 2, 4] if x[field] == x[field] and x[field].strip() and x[field].strip() != '-'
                           for iauto in range(len(x["predicted_pairs"]))]) if len(x["predicted_pairs"]) > 0 and any(
        (x[field] == x[field] and x[field].strip() and x[field].strip() != '-' for field in [0, 2, 4])) else -1, axis=1)
    eval_data["rouges_aspectsLr"] = eval_data.apply(
        lambda x: max([rouge.get_scores(x["predicted_pairs"][iauto][1], re.sub("(#|@)", "", x[field].lower()))[0]['rouge-l']["r"]
                       if len(x["predicted_pairs"][iauto][1]) > 0 else 0
                       for field in [1, 3, 5] if x[field] == x[field] and x[field].strip() and x[field].strip() != '-'
                           for iauto in range(len(x["predicted_pairs"]))]) if len(x["predicted_pairs"]) > 0 and any(
        (x[field] == x[field] and x[field].strip() and x[field].strip() != '-' for field in [1, 3, 5])) else -1, axis=1)
    eval_data["rouges_targetsLf"] = eval_data.apply(
        lambda x: max([rouge.get_scores(x["predicted_pairs"][iauto][0], re.sub("(#|@)", "", x[field].lower()))[0]['rouge-l']["f"]
                       if len(x["predicted_pairs"][iauto][0]) > 0 else 0
                       for field in [0, 2, 4] if x[field] == x[field] and x[field].strip() and x[field].strip() != '-'
                           for iauto in range(len(x["predicted_pairs"]))]) if len(x["predicted_pairs"]) > 0 and any(
        (x[field] == x[field] and x[field].strip() and x[field].strip() != '-' for field in [0, 2, 4])) else -1, axis=1)
    eval_data["rouges_aspectsLf"] = eval_data.apply(
        lambda x: max([rouge.get_scores(x["predicted_pairs"][iauto][1], re.sub("(#|@)", "", x[field].lower()))[0]['rouge-l']["f"]
                       if len(x["predicted_pairs"][iauto][1]) > 0 else 0
                       for field in [1, 3, 5] if x[field] == x[field] and x[field].strip() and x[field].strip() != '-'
                           for iauto in range(len(x["predicted_pairs"]))]) if len(x["predicted_pairs"]) > 0 and any(
        (x[field] == x[field] and x[field].strip() and x[field].strip() != '-' for field in [1, 3, 5])) else -1, axis=1)
    eval_data["predicted_pairs_target_tokens"] = eval_data.apply(
        lambda x: [x["predicted_pairs"][iauto][0].split(" ") for iauto in range(len(x["predicted_pairs"]))], axis=1)
    eval_data["target_tokens"] = eval_data.apply(
        lambda x: [re.sub("(#|@)", "", x[field].lower()).split(" ")
                   if x[field] == x[field] and x[field].strip() and x[field].strip() != '-' else []
                   for field in [0, 2, 4]], axis=1)
    eval_data["predicted_pairs_aspect_tokens"] = eval_data.apply(
        lambda x: [x["predicted_pairs"][iauto][1].split(" ") for iauto in range(len(x["predicted_pairs"]))], axis=1)
    eval_data["aspect_tokens"] = eval_data.apply(
        lambda x: [re.sub("(#|@)", "", x[field].lower()).split(" ")
                   if x[field] == x[field] and x[field].strip() and x[field].strip() != '-' else []
                   for field in [1, 3, 5]], axis=1)
    eval_data["target_jaccard"] = eval_data.apply(lambda x: max([1.0 * len(
        set(x["predicted_pairs_target_tokens"][iauto]).intersection(set(true_tokens))) / len(
        set(list(chain.from_iterable([x["predicted_pairs_target_tokens"][iauto], true_tokens]))))
            for true_tokens in x["target_tokens"] for iauto in range(len(x["predicted_pairs"]))])
            if len(x["predicted_pairs"]) > 0 and any(
        (x[field] == x[field] and x[field].strip() and x[field].strip() != '-' for field in [0, 2, 4])) else -1, axis=1)
    eval_data["aspect_jaccard"] = eval_data.apply(lambda x: max([1.0 * len(
        set(x["predicted_pairs_aspect_tokens"][iauto]).intersection(set(true_tokens))) / len(
        set(list(chain.from_iterable([x["predicted_pairs_aspect_tokens"][iauto], true_tokens]))))
            for true_tokens in x["aspect_tokens"] for iauto in range(len(x["predicted_pairs"]))])
            if len(x["predicted_pairs"]) > 0 and any(
        (x[field] == x[field] and x[field].strip() and x[field].strip() != '-' for field in [1, 3, 5])) else -1, axis=1)

    print("Target extraction:")
    print("target_jaccard", np.mean(eval_data[eval_data["target_jaccard"] >= 0]["target_jaccard"].tolist()))
    print("target_partial_match",
          1.0 * sum(eval_data["target_partial_match"].tolist()) / eval_data[eval_data["target_exists"]].shape[0])
    print("target_exact_match",
          1.0 * sum(eval_data["target_exact_match"].tolist()) / eval_data[eval_data["target_exists"]].shape[0])
    print("rouges_targetsLp", np.mean(eval_data[eval_data["rouges_targetsLp"] >= 0]["rouges_targetsLp"].tolist()))
    print("rouges_targetsLr", np.mean(eval_data[eval_data["rouges_targetsLr"] >= 0]["rouges_targetsLr"].tolist()))
    print("rouges_targetsLf", np.mean(eval_data[eval_data["rouges_targetsLf"] >= 0]["rouges_targetsLf"].tolist()))
    print()
    print("Aspect extraction:")
    print("aspect_jaccard", np.mean(eval_data[eval_data["aspect_jaccard"] >= 0]["aspect_jaccard"].tolist()))
    print("aspect_partial_match",
          1.0 * sum(eval_data["aspect_partial_match"].tolist()) / eval_data[eval_data["aspect_exists"]].shape[0])
    print("aspect_exact_match",
          1.0 * sum(eval_data["aspect_exact_match"].tolist()) / eval_data[eval_data["aspect_exists"]].shape[0])
    print("rouges_aspectsLp", np.mean(eval_data[eval_data["rouges_aspectsLp"] >= 0]["rouges_aspectsLp"].tolist()))
    print("rouges_aspectsLr", np.mean(eval_data[eval_data["rouges_aspectsLr"] >= 0]["rouges_aspectsLr"].tolist()))
    print("rouges_aspectsLf", np.mean(eval_data[eval_data["rouges_aspectsLf"] >= 0]["rouges_aspectsLf"].tolist()))
    print()


if __name__ == "__main__":
    reference_collection_name = sys.argv[1]
    test_collection_name = sys.argv[2]
    annotations_path = sys.argv[3] if len(sys.argv)>3 else ""
    with TargetAspectExtractor() as extractor:
        extractor.get_targets_and_aspects(collection_name=reference_collection_name,
            collection_name_test=test_collection_name)
        if annotations_path:
            extractor.evaluate_collection(ground_truth_annotation_path=annotations_path, collection_name_test=test_collection_name)
