import re
import random
import json

span_length = 20

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

                random.shuffle(keywords_list)
        
                # The first keyword encountered
                for word in keywords_list:
                    
                    separate_word = " " + word + " "

                    if separate_word in line:
                        line_list = re.split(r'\\n|\n| ' , line)
                        line_list = [x for x in line_list if not x.endswith(":")]
    
                        index = line_list.index(word)

                        # Randomized span of 15 where the word is a a part
                        start_nr = random.randint(1,7)
                        end_nr = span_length - start_nr
                        start_index = index - start_nr
                        extracted_sentence = line_list[start_index:index + end_nr]

                        str1 = " " 
                        text_sample = str1.join(extracted_sentence)

                        jsonl_line = {"thread":thread ,"id":id_nr,"text":text_sample,"keyword": separate_word ,"starting_index":start_index,"span_length":span_length}

                        f.write(json.dumps(jsonl_line,ensure_ascii=False) + "\n")

                        break

        
                id_nr += 1



feminism_thread = "politics-feminism"
#rasforskning_thread = "science-biologi-rasforskning"
#prostitution_etik_moral_och_politik_thread = "mobility-bilar-prostitution_etik_moral_och_politik"
#antisemitism_sionism_och_judiska_maktforhallanden_thread = "rest-arkiverade_forum-antisemitism_sionism_och_judiska_maktforhallanden"
#jamstalldhet_och_diskriminering_thread = "society-jamstalldhet_och_diskriminering"
#terrorism_thread = "politics-terrorism"
#integration_och_invandring_thread = "politics-integration_och_invandring"
nationalsocialism_fascism_och_nationalism_thread = "politics-nationalsocialism_fascism_och_nationalism"

#extract_keyword_json("./sampled_data/thread_text_files/feminism.txt","./sampled_20_data/keyword_data/keyword_feminism.jsonl",feminism_thread)

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

    
                if len(text) >= span_length:
                    
                    f.write(json.dumps(result,ensure_ascii=False) + "\n")


#clean_data("sampled_20_data/keyword_data/500_each_400prostetmoral_600rasforsk__keyword_20span_dataset.jsonl","./sampled_20_data/keyword_data/clean_500_each_400prostetmoral_600rasforsk__keyword_20span_dataset.jsonl")

def createDataset(input_txt,keyword_name,keyword_final_name,random_name,random_final_name, thread):


    #extract_keyword_json(input_txt,keyword_name, thread)           
    extract_random_json(input_txt,random_name, thread)
    #clean_data(keyword_name,keyword_final_name)
    clean_data(random_name,random_final_name)

feminism_thread = "politics-feminism"
#rasforskning_thread = "science-biologi-rasforskning"
#prostitution_etik_moral_och_politik_thread = "mobility-bilar-prostitution_etik_moral_och_politik"
#antisemitism_sionism_och_judiska_maktforhallanden_thread = "rest-arkiverade_forum-antisemitism_sionism_och_judiska_maktforhallanden"
#jamstalldhet_och_diskriminering_thread = "society-jamstalldhet_och_diskriminering"
#terrorism_thread = "politics-terrorism"
#integration_och_invandring_thread = "politics-integration_och_invandring"
#nationalsocialism_fascism_och_nationalism_thread = "politics-nationalsocialism_fascism_och_nationalism"


# createDataset(
#     './sampled_data/thread_text_files/nationalsocialism_fascism_och_nationalism.txt',
#     './keyword_nationalsocialism_fascism_och_nationalism.jsonl',
#     './sampled_20_data/keyword_nationalsocialism_fascism_och_nationalism.jsonl',


#     './sampled_20_data/random_data/random_feminism.jsonl',
#     './sampled_20_data/random_feminism.jsonl',
#     feminism_thread
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
                keyword = result['keyword']
                starting_index = result["starting_index"]
                span_length = result["span_length"]


                jsonl_line = {"id": it,"thread":thread ,"thread_id":id_nr,"text":text_sample,"keyword": keyword ,"starting_index":starting_index,"span_length":span_length}

                f.write(json.dumps(jsonl_line,ensure_ascii=False) + "\n")

                it += 1


#reformat_final_data("./sampled_20_data/keyword_data/500_each_400prostetmoral_600rasforsk__keyword_20span_dataset.jsonl","./sampled_20_data/keyword_20span_dataset.jsonl.jsonl")

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


shuffle_jsonl("./sampled_20_data/keyword_20span_dataset.jsonl.jsonl","./sampled_20_data/shuffled_keyword_20span_dataset.jsonl.jsonl")