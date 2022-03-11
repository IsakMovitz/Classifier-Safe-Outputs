import re
import random
import json

span_length = 15

# Not sure whats wrong but maybe just make a cleaning script for the jsonl file ? 
def extract_random_json(input_file,output_file,thread):

    id_nr = 0

    with open(input_file) as current_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in current_file:       # Right now only taking one sentence from each thread.

                # The first keyword encountered
                
                line_list = re.split(r'\\n|\n| ' , line)
                line_list = [x for x in line_list if not x.endswith(":")]

                length = len(line_list)

                if length > span_length:
                   

                    starting_index = random.randrange(0,length - span_length)
                    extracted_sentence = line_list[starting_index:starting_index + span_length]

                    str1 = " " 
                    text_sample = str1.join(extracted_sentence)

                    jsonl_line = {"thread":thread ,"id":id_nr,"text":text_sample,"starting_index":starting_index,"span_length":span_length}

                    f.write(json.dumps(jsonl_line,ensure_ascii=False) + "\n")


                id_nr += 1


def extract_keyword_json(input_file,output_file,thread):
    #Keywords
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

                # Removing weirdness with spaces being in lists, now all spans will be 15
                while '' in text:
                    text.remove('')

                if len(text) >= 15:
                    
                    f.write(json.dumps(result,ensure_ascii=False) + "\n")
              
def createDataset(input_txt,keyword_name,keyword_final_name,random_name,random_final_name, thread):


    extract_keyword_json(input_txt,keyword_name, thread)           
    extract_random_json(input_txt,random_name, thread)
    clean_data(keyword_name,keyword_final_name)
    clean_data(random_name,random_final_name)

# rasforskning_thread = "science-biologi-rasforskning"
# prostitution_etik_moral_och_politik_thread = "mobility-bilar-prostitution_etik_moral_och_politik"
# feminism_thread = "politics-feminism"
# antisemitism_sionism_och_judiska_maktforhallanden_thread = "rest-arkiverade_forum-antisemitism_sionism_och_judiska_maktforhallanden"
# jamstalldhet_och_diskriminering_thread = "society-jamstalldhet_och_diskriminering"
# terrorism_thread = "politics-terrorism"
# integration_och_invandring_thread = "politics-integration_och_invandring"
# nationalsocialism_fascism_och_nationalism_thread = "politics-nationalsocialism_fascism_och_nationalism"

# createDataset(
#     './thread_text_files/nationalsocialism_fascism_och_nationalism.txt',
#     './keyword_nationalsocialism_fascism_och_nationalism.jsonl',
#     './new_clean_data/keyword_nationalsocialism_fascism_och_nationalism.jsonl',
#     './random_nationalsocialism_fascism_och_nationalism.jsonl',
#     './new_clean_data/random_nationalsocialism_fascism_och_nationalism.jsonl',
#     nationalsocialism_fascism_och_nationalism_thread
# )

def reformat_final_data(input_file, output_file):
    with open(input_file, 'r') as json_file:
        json_list = list(json_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            
            it = 0
            for json_str in json_list:
                result = json.loads(json_str)

                thread = result["thread"]
                id_nr = result["id"]
                text_sample = result["text"]
                starting_index = result["starting_index"]
                span_length = result["span_length"]


                jsonl_line = {"id": it,"thread":thread ,"thread_id":id_nr,"text":text_sample,"starting_index":starting_index,"span_length":span_length}

                f.write(json.dumps(jsonl_line,ensure_ascii=False) + "\n")

                it += 1


#reformat_final_data("./clean_data/final_dataset.jsonl","./clean_data/reformatted_final_dataset.jsonl")

#clean_data("./keyword_data/flashback_keyword_data.jsonl", "./keyword_data/flashback_keyword_data.jsonl")

#reformat_final_data("./keyword_data/flashback_keyword_data.jsonl","./keyword_data/reformatted_clean_flashback_keyword_data.jsonl")


import random

def shuffle_jsonl(input_file,output_file):
  
    with open(input_file, 'r') as json_file:
        json_list = list(json_file)
        random.shuffle(json_list)
        with open(output_file, 'w', encoding='utf-8') as f:

            for json_str in json_list:
                result = json.loads(json_str)
    
                f.write(json.dumps(result,ensure_ascii=False) + "\n")


shuffle_jsonl("./keyword_data/flashback_keyword_dataset.jsonl","./keyword_data/shuffled_flashback_keyword_dataset.jsonl")