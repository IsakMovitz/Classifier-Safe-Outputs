import json

def clean_annotation(input_filename, output_filename):
    cleaned_data = []

    id_list = []

    #Open annotation file from Prodigy:
    with open(input_filename, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        line_dict = json.loads(json_str)
        id_nr = line_dict['id']
        if id_nr not in id_list:
            id_list.append(id_nr)
            
            text_sample = line_dict['text']
            thread = line_dict['thread']
            thread_id = line_dict['thread_id']
            keyword =  line_dict['keyword'].strip(" ")
            starting_index = line_dict['starting_index']
            span_length = line_dict['span_length']

            # For extra with franscesca

            toxic = line_dict['TOXIC']
        
            if line_dict['answer'] == 'accept': # 1
                cleaned_data.append({"id":id_nr,"thread":thread,"thread_id":thread_id, "text":text_sample, "keyword": keyword ,"starting_index":starting_index,
                "span_length":span_length,"TOXIC":toxic, "TOXIC_2": 1})
            else:
                cleaned_data.append({"id":id_nr,"thread":thread,"thread_id":thread_id,"text":text_sample,"keyword": keyword,"starting_index":starting_index,
                "span_length":span_length,"TOXIC":toxic, "TOXIC_2": 0})

    #Save cleaned back to jsonl format:
    with open(output_filename, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                if item != cleaned_data[-1]:
                    f.write(json.dumps(item,ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(item,ensure_ascii=False))


#clean_annotation("./NEW_DATA/Annotated_data/20span_teach_1000_flashback_keyword_dataset.jsonl","./NEW_DATA/Clean_annotated_data/clean_20span_teach_1000_flashback_keyword_dataset.jsonl")

# Franscesca

clean_annotation("./NEW_DATA/Annotated_data/franscesca_test_dataset.jsonl","./NEW_DATA/Clean_annotated_data/clean_franscesca_test_dataset.jsonl" )


