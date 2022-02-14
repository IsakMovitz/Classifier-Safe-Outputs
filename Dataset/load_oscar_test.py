from datasets import load_dataset

'''
Source: https://huggingface.co/docs/datasets/installation.html
Source : https://huggingface.co/datasets/oscar-corpus/OSCAR-2109

Login to authenticate and access the dataset
huggingface-cli login 

'oscar-corpus/OSCAR-2109', dataset
'oscar'
'deduplicated_sv', configuration (subset) swedish

Apart from name and split, the datasets.load_dataset() 
method provide a few arguments which can be used to control where the data is cached (cache_dir), 
some options for the download process it-self like the proxies and whether the download cache should 
be used (download_config, download_mode)'''


### Download the whole dataset ###
#oscar_dataset = load_dataset('oscar-corpus/OSCAR-2109', 'deduplicated_sv', use_auth_token=True)

#train = oscar_dataset["train"]
#test = oscar_dataset["test"]
#print(train[0])


### Streaming over the dataset so you dont have to download the whole thing ###
'''first parameter: dataset
second parameter: configuration (subset of dataset)
third parameter:  'train' = The full `train` split.
fourth parameter: streaming or not
fifth paramete: authentication

Dataset Object
'id'
'text'
'meta'

'''
dataset = load_dataset('oscar-corpus/OSCAR-2109', "deduplicated_sv", split='train', streaming=True, use_auth_token=True)
dataset_iterator = iter(dataset)

#for i in range(1):      # Get the n first objects of the dataset streamed 
dataobject = next(dataset_iterator)
object_id = dataobject['id']
text = dataobject['text']       # string type

sentence_list = text.split('.')

print(sentence_list[0])
print(sentence_list[1])