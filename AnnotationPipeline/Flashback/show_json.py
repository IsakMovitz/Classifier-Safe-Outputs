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


# Subtopic of dataset
rasforskning_file = "./subtopics_files/rasforskning.json"



data = ijson.parse(open(rasforskning_file,'r'),multiple_values=True)




range_of_instances = 100
it = 0
for prefix, event, value in data:

    it += 1
 
    if it <= range_of_instances:

        print(value)
        #print(str(prefix) + "   " + str(event) + "   " + str(value))
          
    else:
        break
