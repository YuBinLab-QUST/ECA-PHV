# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:43:00 2019


"""
import re,os,sys,csv
from collections import Counter
pPath = re.sub(r'AAC$', '', os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)
import readFasta
import pandas as pd

def AAC(fastas, **kw):
	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		count = Counter(sequence)
		for key in count:
			count[key] = count[key]/len(sequence)
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)
	return encodings

fastas = readFasta.readFasta("")
kw=  {'path': r"AAC",'train':r"human1.txt",'order':'ARNDCQEGHILKMFPSTWYVX'}
data_AAC=AAC(fastas, **kw)
#AAC=data_AAC.to_list
AAC=pd.DataFrame(data=data_AAC)
AAC.to_csv('')