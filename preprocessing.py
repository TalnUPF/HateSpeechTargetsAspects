import pandas as pd
import re
import os

metadata = [
    {"filename": "data/all_data_waseem.csv",
     "sep": ",",
     "no_header": False,
     "textfield": "text",
     "classfield": "Class",
     "equal": True,
     "classes": ["sexism", "racism"],
     "special_replacements": [["((&lt;3)|(&lt;(-)*)|((-)*&gt;)|(#?mkr(\.)?)|(#?MKR(\.)?)|(#?Mkr(\.)?))", ""],
                              ["&amp;", "and"]],
     "output_names": ["waseem_sexism.txt", "waseem_racism.txt"]}]

output_folder = "processed_data/clean_texts"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def capswords_to_lower(text):
    caps = re.finditer("[A-Z][A-Z]+", text)
    caps_list = [[cap.start(), cap.end(), cap.group(0).lower()] for cap in caps]
    for cap in reversed(caps_list):
        text = text[:cap[0]] + cap[2] + text[cap[1]:]
    return text


def mentions_hashtags_to_words_with_dot(text):
    caps = re.finditer("(@|#)[^\s]+", text)
    caps_list = [[cap.start(), cap.end(), cap.group(0)] for cap in caps]
    for cap in reversed(caps_list):
        if cap[2][-1] != ":":
            text = text[:cap[0]] + cap[2][1:] + "." + text[cap[1]:]
    return text


for hate_data in metadata:
    if hate_data["no_header"]:
        hate = pd.read_csv(hate_data["filename"], sep=hate_data["sep"], header=None)
    else:
        hate = pd.read_csv(hate_data["filename"], sep=hate_data["sep"])
    hate["clean"] = hate[hate_data["textfield"]].apply(lambda x: re.sub("^RT\s*", "", x))
    hate["clean"] = hate["clean"].apply(lambda x: mentions_hashtags_to_words_with_dot(x))
    hate["clean"] = hate["clean"].apply(lambda x: re.sub("http[^\s]+", "", re.sub("@", "", x)))
    for replacement in hate_data["special_replacements"]:
        hate["clean"] = hate["clean"].apply(lambda x: re.sub(replacement[0], replacement[1], x))
    hate["clean"] = hate["clean"].apply(
        lambda x: re.sub("^\s*(,|\.)+\s*", "", re.sub("(\r|\n)+", " ", re.sub("#", ", ", x))))

    hate["clean"] = hate["clean"].apply(lambda x: capswords_to_lower(x))
    hate["clean"] = hate["clean"].apply(lambda x: re.sub(u"\u2026", "...", x))

    for dclass, hate_class in enumerate(hate_data["classes"]):
        if hate_data["equal"]:
            res = hate[(hate[hate_data["classfield"]] == hate_class)]
        else:
            res = hate[(hate[hate_data["classfield"]] != hate_class)]
        with open(os.path.join(output_folder, hate_data["output_names"][dclass]), "wb") as fout:
            fout.write(("\n".join(res["clean"].tolist()) + "\n").encode("utf8"))
