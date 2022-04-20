import json  
def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)


input_filename = './EXTENDED_CLEAN_INTER.jsonl'


with open(input_filename, 'r') as json_file:
        json_list = list(json_file)

annotator_1 = []
annotator_2 = []

for json_str in json_list:
    line_dict = json.loads(json_str)

    
    toxic_1 = line_dict['TOXIC']
    

    toxic_2 = line_dict['TOXIC_2']

    if toxic_2 != "ignored":

        annotator_1.append(toxic_1)
        annotator_2.append(toxic_2)

print(len(annotator_1))
kappa_value = cohen_kappa(annotator_1, annotator_2)
print(kappa_value)