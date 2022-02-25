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
# oscar_dataset = load_dataset('oscar-corpus/OSCAR-2109', 'deduplicated_sv', use_auth_token=True)
# training_data = oscar_dataset["train"] 

'''
id: id of the instance
text: actual textdata to be classified
start_index: from what index of words in the raw data that the span is taken
span_length: the length of the span extracted from a single raw data index

Take a random start index within the raw_text span.
Between index 0 -> (N - 15) , N being length of raw text in words

What to do if the length of an id of raw_text is < 15 words? 
Right now just ignore it

'''
### Parsing into JSONL format from Huggingface ###

def parse_to_jsonl(data,span_length,filename,nr_samples):
    random_samples = []    
    id_nr = -1
    for i in range(nr_samples):
        id_nr += 1 
        instance = data[i]
        instance_text = instance['text']
        instance_text_list = instance_text.split(' ')

        length = len(instance_text_list)
        if length > span_length:
            starting_index = random.randrange(0,length - span_length)
            random_sample_list = instance_text_list[starting_index:starting_index + span_length]

            # Back to string
            str1 = " " 
            text_sample = str1.join(random_sample_list)
            random_samples.append({"id":id_nr,"text":text_sample,"starting_index":starting_index,"span_length":span_length})
        
        else:
            id_nr = id_nr - 1

    with open(filename, 'w', encoding='utf-8') as f:
        for item in random_samples:
            if item != random_samples[-1]:
                f.write(json.dumps(item,ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(item,ensure_ascii=False))

#parse_to_jsonl(training_data,15,"Data/raw_racism_data_10.jsonl",10)  # Since index 41 is < 15 in length

### Sampling spans from a jsonl file with id,text ###

def sample_span_from_jsonl(input_filename,output_filename,span_length):

    with open(input_filename, 'r') as json_file:
            json_list = list(json_file)

    random_samples = []
    for json_str in json_list:
        line_dict = json.loads(json_str)

        id_nr = line_dict['id']
        instance_text = line_dict['text']
        instance_text_list = instance_text.split(' ')

        length = len(instance_text_list)
        if length > span_length:
            starting_index = random.randrange(0,length - span_length)
            random_sample_list = instance_text_list[starting_index:starting_index + span_length]

            # Back to string
            str1 = " " 
            text_sample = str1.join(random_sample_list)
            random_samples.append({"id":id_nr,"text":text_sample,"starting_index":starting_index,"span_length":span_length})
        
    with open(output_filename, 'w', encoding='utf-8') as f:
            for item in random_samples:
                f.write(json.dumps(item,ensure_ascii=False) + "\n")

#sample_span_from_jsonl('Data/pre_search_racism_whole.jsonl','Data/samples_racism.jsonl',15)