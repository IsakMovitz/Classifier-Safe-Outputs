import pandas as pd
import json

json_file_name = "./test.json"

flashback_file = "./flashback.json"

#df = pd.read_json(flashback_file)      # memory error



def read_in_chunks():
    chunks = pd.read_json(flashback_file, lines=True, chunksize = 100)

    print(type(chunks))

    outer_subject_list = []
    inner_subject_list = []
    it_range = 10000000
    it = 0

    with open('newfile.txt', 'w') as f:
        

        for chunk in chunks:

            #it += 1
        
            path = chunk['path']
            text = chunk['text']
            path_data = chunk['path'].iloc[1]
            outer_subject = path_data[0]
            inner_subject = path_data[1]


            #if(it <= it_range):

            if outer_subject not in outer_subject_list:
                print("\n ### " + outer_subject + " ###")
                print("---------------------")
                f.write("\n ### " + outer_subject + " ###\n")
                f.write("---------------------\n")
                
                outer_subject_list.append(outer_subject)
            
            if inner_subject not in inner_subject_list:
                print(inner_subject)
                f.write(inner_subject + "\n")
                inner_subject_list.append(inner_subject)

        # else:

        #     break


read_in_chunks()


 # Formats for json : 'records' , 'index' , 'split' , 'table' , and 'values' 


def output_sample():
    chunks = pd.read_json(flashback_file, lines=True, chunksize = 10)

    # Formats for json : 'records' , 'index' , 'split' , 'table' , and 'values'
    filetype = 'table'

    output_filename = "./" + filetype + "_sample.json"

    it_range = 2
    it = 0

    with open(output_filename, 'w', encoding='utf-8') as f:
        for chunk in chunks:

            it += 1
            result = chunk.to_json(orient= filetype)
            parsed = json.loads(result)
            
            if(it <= it_range):
                    f.write(json.dumps(parsed,ensure_ascii=False))
            else:

                break


#output_sample()