from datasets import load_dataset

oscar_dataset = load_dataset('oscar-corpus/OSCAR-2109', 'deduplicated_sv', use_auth_token=True)

'''
Only including training data.

DatasetDict({
    train: Dataset({
        features: ['id', 'text', 'meta'],
        num_rows: 10164521
    })
})

We want something in the format of (Maybe create a csv file?):

ID , TEXT , CLASS

Ill start with just 15 words per sentence as my basis for dividing the data.
Probably need to clean it somehow though?'''

training_data = oscar_dataset["train"] 

first_instance = training_data[0]
first_instance_text = first_instance['text']

first_instance_text_list = first_instance_text.split(' ')

print(text_list)

print(len(text_list))

# for i in range(3):
#     print(training_data[i])
#     print("#########################################################")

