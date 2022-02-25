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

### Parsing into JSONL format ###

def search_and_parse_to_jsonl(data,filename,nr_samples):

    #keywords = ["invandrare","muslim","nmr","jude","judar","judisk", "neger","blatte","etnisk"]
    keywords = ["jävla","jävel","idiot","dum","korkad","hatar"]


    samples = []    
    id_nr = -1
    for i in range(nr_samples):
        id_nr += 1 
        instance = data[i]
        instance_text = data[i]['text']
        instance_text_list = instance_text.split(' ')


        if(any(x in keywords for x in instance_text_list)):
            
            samples.append({"id":id_nr,"text":instance_text})
    

    with open(filename, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item,ensure_ascii=False) + "\n")
    
             

#search_and_parse_to_jsonl(training_data,"Data/pre_search_threat_million.jsonl",1000000)