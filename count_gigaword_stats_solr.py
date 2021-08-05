import os
import pandas as pd
import requests
import pickle
import math


def count_tf_idf(in_topfolder, out_topfolder):
    if not os.path.exists(out_topfolder):
        os.makedirs(out_topfolder)

    for infolder in os.listdir(in_topfolder):
        conc_folderin = os.path.join(in_topfolder, infolder)
        conc_folderout = os.path.join(out_topfolder, infolder)
        if not os.path.exists(conc_folderout):
            os.makedirs(conc_folderout)
        conc_df_files = os.listdir(conc_folderin)
        cocn_doc_num_dict = {}
        dictpickle = os.path.join("processed_data", "hate_speech_pos_stats_solr_dict.pickle")
        Ndoc = 161105350
        if os.path.exists(dictpickle):
            cocn_doc_num_dict = pickle.load(open(dictpickle, "rb"))
        for conc_df_file in conc_df_files:
            df = pd.read_csv(os.path.join(conc_folderin, conc_df_file))
            concepts_set_file = conc_df_file[:-4] + "_concepts_set.pickle"
            if not os.path.exists(concepts_set_file):
                concepts_set = list(set([conc for conc in df["concept"].tolist() if conc and conc == conc]))
                pickle.dump(concepts_set, open(concepts_set_file, "wb"))
            else:
                concepts_set = pickle.load(open(concepts_set_file, "rb"))
            print(len(concepts_set))
            for dconc, conc in enumerate(concepts_set):
                if dconc % 100 == 0:
                    print(dconc)
                    pickle.dump(dict(cocn_doc_num_dict), open(dictpickle, "wb"))
                if conc in cocn_doc_num_dict:
                    continue
                res = requests.get(
                    "http://clasificador-taln.upf.edu/index/english_giga/select?q=text:\"%20" +
                    conc.replace(" ", "%20").replace("%", "%25") + "%20\"").json()
                if "response" in res and "numFound" in res["response"]:
                    cocn_doc_num_dict[conc] = res["response"]["numFound"]
            df["DF"] = df["concept"].apply(lambda x: cocn_doc_num_dict[x] if x in cocn_doc_num_dict else 0)
            df["TF-IDF"] = df.apply(
                lambda x: 1.0 * math.log(1.0 + x["tf_collection"]) * (math.log(1.0 * Ndoc / (1.0 + x["DF"])) + 1),
                axis=1)
            df = df.sort_values("TF-IDF", ascending=False)
            df.to_csv(os.path.join(conc_folderout, conc_df_file[:-4] + "_tf_idf.csv"), index=False)


if __name__ == "__main__":
    in_topfolder = os.path.join("processed_data", "reference_concepts_followed_by_verb_not_aux")
    out_topfolder = os.path.join("processed_data", "gigaword_stats")
    count_tf_idf(in_topfolder, out_topfolder)
