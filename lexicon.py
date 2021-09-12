# coding=utf-8

##############################
#### Skript um das Lexikon ###
########## zu erzeugen #######
##############################

import csv
import json

with open('german.csv', mode="r") as infile:
    reader = csv.reader(infile)
    mydict = {rows[0]: rows[1] for rows in reader}

    # print(mydict)

with open('result.json', 'w') as fp:
    json.dump(mydict, fp, ensure_ascii=True)


with open('result.json') as f:
    data = json.load(f)
    print(data)
    # if 'ruhelos' in data.keys():
    #     print('yes')
    data["herrschen"] = "0"
    data["profitiren"] = "1"
    data['Krise'] = "-1"

    # tmp = data["Katastrophe"]

with open("result.json", "w") as jsonFile:
    json.dump(data, jsonFile, ensure_ascii=True)

# print(python_dict["spinnen"])
