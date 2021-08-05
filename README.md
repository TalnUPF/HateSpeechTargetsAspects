# HateSpeechTargetsAspects

## Overview
This repository contains the code and the annotations for the paper "Targets and Aspects in Social Media Hate Speech".

Citation: \
Shvets A, Fortuna P, Soler-Company J, Wanner L. 2021. Targets and Aspects in Social Media Hate Speech. In Proceedings of the 5th Workshop on Online Abuse and Harms (WOAH 2021) at ACL-IJCNLP 2021, pp. 179-190.

## Inference and Evaluation
Run the following command to repeat the evaluation provided in the paper:
```
python evaluation_woah.py
```

## Application to a new dataset
Step 1. Install concept extractor from https://github.com/TalnUPF/ConceptExtraction/

Step 2. Run preprocessing separately on the reference data and a test set:
```
python preprocessing.py
```

Step 3. Apply concept extraction to each set of texts:
```
python run_concept_extraction.py --input-file-path path_to_a_file --save-tokenized-texts

```

Step 4. Run the following commands only on the reference data:
```
python list_concepts_hate_speech.py

count_gigaword_stats_solr.py
```

Step 5. Extract targets and aspects from arbitrary posts (test set):
```
python target_aspect_extractor.py reference_collection_name test_collection_name
```

Step 6. For evaluation, add a path to ground truth data as a third parameter:
```
python target_aspect_extractor.py reference_collection_name test_collection_name path_to_ground_truth_data
```
