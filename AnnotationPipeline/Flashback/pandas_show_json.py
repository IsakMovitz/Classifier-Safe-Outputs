import pandas as pd
import json

json_file_name = "./test.json"

flashback_file = "./flashback.json"

rasforskning_file = "./subtopics_files/rasforskning.json"

aktuella_file = "./topics_files/aktuella_brott_och_kriminalfall.json"

organiserat_brott_file = "./topics_files/organiserad_brottslighet.json"


# with open(rasforskning_file) as f:
#    data = json.load(f)

# print(pd.DataFrame(data))
#df = pd.read_json(flashback_file)      # memory error


# Förfiltrera data 
# q to exit 

# jq '' flashback.json | less
# jq '.path' flashback.json | less
# jq '.path[1]' flashback.json | less
# jq '.path' flashback.json > output.txt

# jq '.path[1] | select(.location=="Stockholm")' json

# jq 'select(.path[0]=="politics")' flashback.json | less
# jq 'select(.path[2]=="sexnoveller")' flashback.json | less
# jq 'select(.path[2]=="spel_support")' flashback.json | less
# jq 'select(.path[1]=="arkiverade_forum")' flashback.json > arkiverade_forum.json

# jq 'select(.path[1]=="organiserad_brottslighet")' flashback.json > organiserad_brottslighet.json
# jq 'select(.path[1]=="aktuella_brott_och_kriminalfall")' flashback.json > aktuella_brott_och_kriminalfall.json

# jq 'select(.path[2]=="rasforskning")' flashback.json > rasforskning.json
# jq 'select(.path[1]=="rasforskning")' flashback.json > rasforskning.json

# jq 'select(.path[1]=="biologi")' flashback.json | less
# jq 'select(.path[2]=="rasforskning")' flashback.json > rasforskning.json

# jq 'select(.path[2]=="antisemitism_sionism_och_judiska_maktforhallanden")' ./topics_files/arkiverade_forum.json | less



def print_chunks():
    chunks = pd.read_json(organiserat_brott_file, lines=True, chunksize = 100, orient='records')

    print(type(chunks))


    it_range = 10
    it = 0

    
    for chunk in chunks:
        print(chunk)
        it += 1

        # if it < it_range:
            
        #     print(path)
        #     path = chunk['path']
        #     text = chunk['text']
        #     path_data = chunk['path'].iloc[1]
        #     path_length = len(path_data)

        #     outer_subject = path_data[0]
        #     inner_subject = path_data[1]

           


print_chunks()

def read_in_chunks():
    chunks = pd.read_json(rasforskning_file, lines=True, chunksize = 100)

    print(type(chunks))

    outer_subject_list = []
    inner_subject_list = []
    most_inner_subject_list = []
    it_range = 10000000
    it = 0

    with open('full_flashback_topics.txt', 'w') as f:
    
        for chunk in chunks:

            #it += 1
        
            path = chunk['path']
            text = chunk['text']
            path_data = chunk['path'].iloc[1]
            path_length = len(path_data)

            outer_subject = path_data[0]
            inner_subject = path_data[1]

            if path_length > 2:
                most_inner_subject = path_data[2]

            if outer_subject not in outer_subject_list:
                print("\n ### " + outer_subject + " ###")
                print("_______________________")
                # f.write("\n ########### " + outer_subject + " ###########\n")
                # f.write("_________________________________________________________________\n")
                outer_subject_list.append(outer_subject)
                
            if inner_subject not in inner_subject_list:
                print(" \n ** " + inner_subject + "**")
                print("-------------")
                # f.write("\n** " + inner_subject + "** \n")
                # f.write("-------------\n")
                inner_subject_list.append(inner_subject)

            if path_length > 2:
                if most_inner_subject not in most_inner_subject_list:
                    print(most_inner_subject)
                    # f.write(most_inner_subject + "\n")
                    most_inner_subject_list.append(most_inner_subject)


#read_in_chunks()


# def read_in_chunks():
#     chunks = pd.read_json(flashback_file, lines=True, chunksize = 10)

#     print(type(chunks))

#     outer_subject_list = []
#     inner_subject_list = []
#     most_inner_subject_list = []
#     path_length_list = []

#     it_range = 10000
#     it = 0

#     with open('full_flashback_topics.txt', 'w') as f:
        
#         for chunk in chunks:

#             #it += 1

#             path = chunk['path']
#             text = chunk['text']
#             path_data = chunk['path'].iloc[1]
#             path_length = len(path_data)

#             outer_subject = path_data[0]
#             inner_subject = path_data[1]

#             if path_length not in path_length_list:
#                 path_length_list.append(path_length)
#                 print(path_length)


#             # if path_length > 2:
#             #     most_inner_subject = path_data[2]

#             # if(it <= it_range):
#             #     print(chunk)
#             # else:
#             #     break

#             # if outer_subject not in outer_subject_list:
#             #     # print("\n ### " + outer_subject + " ###")
#             #     # print("_______________________")
#             #     f.write("\n ### " + outer_subject + " ###\n")
#             #     f.write("________________________\n")
                
#             #     outer_subject_list.append(outer_subject)
            
#             # if inner_subject not in inner_subject_list:
#             #     # print(" \n ** " + inner_subject + "**")
#             #     # print("-------------")
#             #     f.write("\n** " + inner_subject + "** \n")
#             #     f.write("-------------\n")
#             #     inner_subject_list.append(inner_subject)

#             # if path_length > 2:
#             #     if most_inner_subject not in most_inner_subject_list:
#             #         # print(most_inner_subject)
#             #         f.write(most_inner_subject + "\n")
#             #         most_inner_subject_list.append(most_inner_subject)


#         # else:

#         #     break


# read_in_chunks()


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