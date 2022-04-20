import json  
# 291 / 1000 , for keyword 20 span



# # 

# input_filename = './EXTENDED_CLEAN_INTER.jsonl'


# with open(input_filename, 'r') as json_file:
#         json_list = list(json_file)

# it = 0

# ignored = 0
# same = 0
# different = 0
# total_toxic_1_positives = 0
# total_toxic_2_positives = 0

# print("Ignored examples:")
# for json_str in json_list:
#     line_dict = json.loads(json_str)

#     toxic_1 = line_dict['TOXIC']
#     toxic_2 = line_dict['TOXIC_2']
    
#     if toxic_1 == 1:
#        total_toxic_1_positives += 1

#     if toxic_2 == 1:
#        total_toxic_2_positives += 1


#     if toxic_2 == 'ignored':
#         ignored += 1
#         print(toxic_1)
#         print( "<" + line_dict['text'] + ">")
                
#     elif line_dict['TOXIC'] == line_dict['TOXIC_2']:
#         same += 1
#     else:
#         different += 1   

# print("--------")         
# print("nr_ignored: " + str(ignored))
# print("nr_same: " + str(same))
# print("nr_different: " + str(different))
# print("total_toxic_1_positives: " + str(total_toxic_1_positives))
# print("total_toxic_2_positives: " + str(total_toxic_2_positives))
# print("nr_different: " + str(different))

# proportion = same / (same + different + ignored)
# print("proportion of total agreed: " + str(proportion))




### Counting specific labels ###
# input_filename = './Data/EXTRA_500.jsonl' 
# with open(input_filename, 'r') as json_file:
#         json_list = list(json_file)

# it = 0
# it2 = 0
# for json_str in json_list:
#     line_dict = json.loads(json_str)
#     toxic = line_dict['TOXIC']

#     if toxic == 1:
#         it += 1
#     else:
#         it2 += 1

# print(it)
# print(it2)

# For staple diagram 

with open("./keywords.txt") as file:
    #lines = file.readlines()
    lines = [line.rstrip("\n") for line in file]


keywords = lines # length 37 
amount_per_keyword = []
toxic_amount_per_keyword = []
safe_amount_per_keyword = []

input_filename = './RESHUFFLED_FINAL_20SPAN_KEYWORD_DATASET.jsonl'

with open(input_filename, 'r') as json_file:
        json_list = list(json_file)

for word in keywords: 

    word_count = 0 
    toxic_count = 0
    safe_count = 0
    for json_str in json_list:
        line_dict = json.loads(json_str)

        keyword = line_dict['keyword']
        toxic = line_dict['TOXIC']

        if keyword == word: 
            word_count += 1

            if toxic == 1:
                 toxic_count += 1

            elif toxic == 0:
                safe_count += 1

    amount_per_keyword.append(word_count)
    toxic_amount_per_keyword.append(toxic_count)
    safe_amount_per_keyword.append(safe_count)


keywords[13] = 'n*ger'
import matplotlib.pyplot as plt
# plt.rcParams['xtick.labelsize'] = "large"
labels = keywords
toxic_keywords = toxic_amount_per_keyword
safe_keywords = safe_amount_per_keyword

width = 0.65       # the width of the bars: can also be len(x) sequence

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, ax = plt.subplots(figsize=(12,8))


plt.xticks(rotation=90)


ax.bar(labels, toxic_keywords, width, bottom=safe_keywords,
       label='Toxic')
ax.bar(labels, safe_keywords, width, label='Safe')


ax.set_ylabel('Instances of keyword')
ax.set_title('Histogram of keywords')
ax.legend()

plt.savefig('Staple.png')