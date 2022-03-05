import ijson

'''
I guess this works but would be nicer with pandas to work with the smaller files as well, maybe does not matter
Could also use jq all the way almost, but we want jsonl format in the end.

Only problem now is how exactly to get the toxic data.


Chunks of 9:
   start_map
   map_key
path   start_array
path.item   string
path.item   string
path   end_array
   map_key
text   string
   end_map
'''

# Whole dataset
flashback_file = "./flashback.json"

# Topic of dataset
aktuella_file = "./topics_files/aktuella_brott_och_kriminalfall.json"
jamstalldhet_och_diskriminering_file = "./topics_files/jamstalldhet_och_diskriminering.json"

# Subtopic of dataset
rasforskning_file = "./subtopics_files/rasforskning.json"

keywords_list = []
with open('keywords.txt') as keyword_file:
    for line in keyword_file:
        word = line.strip('\n')
        keywords_list.append(word)

data = ijson.parse(open(jamstalldhet_och_diskriminering_file,'r'),multiple_values=True)

ignore_words_list = []
range_of_instances = 0
it = 0
for prefix, event, value in data:

    it += 1

    if it <= range_of_instances:
        
        if value != None:
            if it <= 7:
                ignore_words_list.append(value)
            elif value not in ignore_words_list:
                print(value)
                
        # if value != None:
        #     #print(value)
        # #print(str(prefix) + "   " + str(event) + "   " + str(value))
          
    else:
        break
