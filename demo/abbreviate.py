import sys
from collections import defaultdict
import json
import re
import pickle

if __name__ == '__main__':
    abbreviation = defaultdict(list)
    abbreviation['EMSPlus'].append(['','emsplus'])
    abbreviation['Director'].append(['','director'])
    for itheme in ['rcp','umac','emsplus','director']:
        with open(f'dataset/zedxs/text/{itheme}/log.txt','r') as fp:
            data = [i.split('~')[1].strip() for i in fp.readlines()[1:]]
        for i in range(0,len(data)-1,2):
            abbreviation[data[i]].append([data[i+1],itheme])
    repp = re.compile('[0-9a-zA-Z]+')
    with open('question.jsonl_bak','r') as fp:
        data = fp.readlines()
        for i in data:
            djson = json.loads(i.strip())
            print(djson)
            setlist = []
            for i in repp.findall(djson['query']):
                if i in abbreviation.keys():
                    themeset = set()
                    print(abbreviation[i])
                    for j in abbreviation[i]: 
                        themeset.add(j[1])
                    setlist.append(themeset)
            if len(setlist)>0:
                theme = set.intersection(*setlist)
                if len(theme)==0:
                    theme = set.union(*setlist)
                print(f'theme:{list(theme)}')

    
    with open('abbreviate.bin','wb') as fp:
        pickle.dump(abbreviation,fp) 
