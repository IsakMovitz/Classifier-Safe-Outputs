import json

def clean_annotation(input_filename, output_filename):
    cleaned_data = []
    #Open annotation file from Prodigy:
    with open("data/" + input_filename, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        line_dict = json.loads(json_str)
        id_nr = line_dict['id']
        text_sample = line_dict['text']
        starting_index = line_dict['starting_index']
        span_length = line_dict['span_length']

        if line_dict['answer'] == 'accept': # 1
            cleaned_data.append({"id":id_nr,"text":text_sample,"starting_index":starting_index,
            "span_length":span_length,"toxic_class":1})
        else:
            cleaned_data.append({"id":id_nr,"text":text_sample,"starting_index":starting_index,
            "span_length":span_length,"toxic_class":0})

    #Save cleaned back to jsonl format:
    with open("data/" + output_filename, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                if item != cleaned_data[-1]:
                    f.write(json.dumps(item,ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(item,ensure_ascii=False))


clean_annotation()