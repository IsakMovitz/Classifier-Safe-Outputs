

# Very similar problem to what you are doing here, with active learning as well.
# https://prodi.gy/docs/text-classification


# START ANNOTATION
# python3 -m prodigy ner.manual ner_news_headlines blank:en ./data/news_headlines.jsonl --label PERSON,ORG,PRODUCT,LOCATION

# LOAD ANNOTATED
# python3 -m prodigy db-out ner_news_headlines > ./annotations_news_headlines.jsonl
# python3 -m prodigy db-out toxic_swe_dataset > ./data/annotated_oscar_15.jsonl


# TESTING STUFF
# python3 -m prodigy ner.manual toxic_swe_dataset blank:en ./oscar_data.jsonl --label TOXIC,SAFE
# python3 -m prodigy ner.manual toxic_swe_dataset blank:en ./oscar_data.jsonl --label TOXIC,SAFE

# recipe binary or mult classifier, textcat.manual currently the best

# python3 -m prodigy textcat.manual language_identification ./web_dump.jsonl --label English
# python3 -m prodigy textcat.manual toxic_data ./data/raw_oscar_15.jsonl --label TOXIC

# Use with : python3 -m prodigy custom-recipe toxic_swe_dataset ./data.jsonl -F prodigy_test.py
# Use with : python3 -m prodigy custom-recipe your_dataset ./data.jsonl -F recipe.py


# CURRENT BEST METHOD

# View all datasets created
# python3 -m prodigy stats -l
# Check how to delete as well

# Loading data into Prodigy
# python3 -m prodigy textcat.manual toxic_binary_15 ./data/raw_oscar_15.jsonl --label TOXIC,SAFE --exclusive
# CURRENT = python3 -m prodigy textcat.manual toxic_binary ./data/raw_oscar_15.jsonl --label TOXIC

# Getting annotated results back from Prodigy to jsonl
# python3 -m prodigy db-out toxic_swe_dataset > ./annotated_oscar_15.jsonl

# python3 -m prodigy textcat.manual toxic_binary ./data/raw_oscar_15.jsonl --label TOXIC

# CURRENT = python3 -m prodigy db-out toxic_binary > ./annotated_oscar_15.jsonl

# Maybe make a script to turn Prodigy output into more clean jsonl? 
