import json  
# 291 / 1000 , for keyword 20 span

input_filename = '../AnnotationPipeline/NEW_DATA/Clean_annotated_data/clean_20span_teach_1000_flashback_keyword_dataset.jsonl'

with open(input_filename, 'r') as json_file:
        json_list = list(json_file)

it = 0
for json_str in json_list:
    line_dict = json.loads(json_str)

    if line_dict['TOXIC'] == 1:
        it += 1

print(it)

### Counting specific labels ###
# it = 0
# for json_str in json_list:
#     line_dict = json.loads(json_str)
#     sex = line_dict['sexually_explicit']

#     if sex == 1:
#         it += 1

# print(it)