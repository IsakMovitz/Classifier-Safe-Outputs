import json  

input_filename = 'Data/test_data.jsonl'

with open(input_filename, 'r') as json_file:
        json_list = list(json_file)

it = 0
for json_str in json_list:
    line_dict = json.loads(json_str)

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