from datasets import load_dataset
import json
import random as random
'''
Only including training data.

DatasetDict({
    train: Dataset({
        features: ['id', 'text', 'meta'],
        num_rows: 10164521
    })
})
'''

### Loading the data ###
oscar_dataset = load_dataset('oscar-corpus/OSCAR-2109', 'deduplicated_sv', use_auth_token=True)

training_data = oscar_dataset["train"] 

# first_instance = training_data[0]
# first_instance_text = first_instance['text']

# first_instance_text_list = first_instance_text.split(' ')

# print(first_instance_text_list)

# length = len(first_instance_text_list)

# print(length)

# a = first_instance_text_list[length - 15:length]

# #first_instance_text_list[start_index:start_index + 15]

# print(a)
'''
id: id of the instance
text: actual textdata to be classified
start_index: from what index of words in the raw data that the span is taken
span_length: the length of the span extracted from a single raw data index

Take a random start index within the raw_text span.
Between index 0 -> (N - 15) , N being length of raw text in words

'''

### Parsing into JSONL format ###

random_five_sample = []    
span_length = 15

for i in range(5):
    instance = training_data[i]
    instance_text = instance['text']
    instance_text_list = instance_text.split(' ')

    length = len(instance_text_list)
    starting_index = random.randrange(0,length - span_length)
    random_sample_list = instance_text_list[starting_index:starting_index + span_length]

    print(random_sample_list)

    # Back to string
    str1 = " " 
    text_sample = str1.join(random_sample_list)
    random_five_sample.append({"id":i,"text":text_sample,"starting_index":starting_index,"span_length":span_length})
    
print(random_five_sample)

# PROBLEMS WITH SWEDISH LETTERS!

with open("oscar_data.jsonl", 'w') as f:
    for item in random_five_sample:
        f.write(json.dumps(item) + "\n")

# with open("oscar_data.jsonl") as f:
#     for line in f:
#         print(line)


