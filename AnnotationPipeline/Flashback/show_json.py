import ijson

json_file_name = "./test.json"

flashback_file = "./flashback.json"

data = ijson.parse(open(flashback_file,'r'),multiple_values=True)

# Chunks of 9:
#    start_map
#    map_key
# path   start_array
# path.item   string
# path.item   string
# path   end_array
#    map_key
# text   string
#    end_map

##########################################

#    start_map   None
#    map_key   path
# path   start_array   None
# path.item   string   data
# path.item   string   it_sakerhet
# path   end_array   None
# text   string   __087:
# end_map   None

#############################
# first 100 000 only data

range_of_instances = 9
it = 0
for prefix, event, value in data:
    it += 1
 
    if it <= range_of_instances:

        if value == 'data':
            print(str(prefix) + "   " + str(event) + "   " + str(value))
            #print()
    else:
        break


# for prefix, the_type, value in ijson.parse(open(flashback_file),multiple_values=True):
#     print(str(prefix) + "   " + str(the_type) + "   " + str(value))



# for i in range(1):


# for prefix, the_type, value in ijson.parse(open(flashback_file),multiple_values=True):
#     print(str(prefix) + "   " + str(the_type) + "   " + str(value))