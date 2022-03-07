import re
import random
import json

#rasforskning_thread = "science-biologi-rasforskning"
#prostitution_etik_moral_och_politik_thread = "mobility-bilar-prostitution_etik_moral_och_politik"
#feminism_thread = "politics-feminism"
#antisemitism_sionism_och_judiska_maktforhallanden_thread = "rest-arkiverade_forum-antisemitism_sionism_och_judiska_maktforhallanden"
#jamstalldhet_och_diskriminering_thread = "society-jamstalldhet_och_diskriminering"
#terrorism_thread = "politics-terrorism"
#integration_och_invandring_thread = "politics-integration_och_invandring"
#nationalsocialism_fascism_och_nationalism_thread = "politics-nationalsocialism_fascism_och_nationalism"


span_length = 15

# Not sure whats wrong but maybe just make a cleaning script for the jsonl file ? 

def extract_json(input_file,output_file,thread):
    # Keywords
    keywords_list = []
    with open('keywords.txt') as keyword_file:
        for line in keyword_file:
            word = line.strip('\n')
            keywords_list.append(word)

    id_nr = 0
    with open(input_file) as current_file:

        with open(output_file, 'w', encoding='utf-8') as f:
            for line in current_file:       # Right now only taking one sentence from each thread.

                # The first keyword encountered
                for word in keywords_list:
                    
                    separate_word = " " + word + " "

                    if separate_word in line:
                        line_list = re.split(r'\\n|\n| ' , line)
                        line_list = [x for x in line_list if not x.endswith(":")]
    
                        index = line_list.index(word)

                        # Randomized span of 15 where the word is a a part
                        start_nr = random.randint(1,7)
                        end_nr = 15 - start_nr
                        start_index = index - start_nr
                        extracted_sentence = line_list[start_index:index + end_nr]

                        str1 = " " 
                        text_sample = str1.join(extracted_sentence)

                        jsonl_line = {"thread":thread ,"id":id_nr,"text":text_sample,"starting_index":start_index,"span_length":span_length}

                        f.write(json.dumps(jsonl_line,ensure_ascii=False) + "\n")

                        break
            
                id_nr += 1



def clean_data(input_file, output_file):

    with open(input_file, 'r') as json_file:
        json_list = list(json_file)
        with open(output_file, 'w', encoding='utf-8') as f:

            for json_str in json_list:
                result = json.loads(json_str)
                text = result['text'].split(" ")

                if len(text) >= 15:
                    
                    f.write(json.dumps(result,ensure_ascii=False) + "\n")



extract_json("./sampled_data/nationalsocialism_fascism_och_nationalism.txt","./sampled_data/nationalsocialism_fascism_och_nationalism.jsonl",nationalsocialism_fascism_och_nationalism_thread)
clean_data("./sampled_data/nationalsocialism_fascism_och_nationalism.jsonl","./sampled_data/clean_data/clean_nationalsocialism_fascism_och_nationalism.jsonl")

