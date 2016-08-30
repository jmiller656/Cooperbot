import re
import numpy as np
import collections


def getText(end=10000):
	f = open('coopertext.txt','r')
	text = ''
	i = 0
	while i<end:
		line = f.readline()
		if not line:
			break
		line = re.sub(' +',' ',line)
		line = re.sub('\n',' ',line)
		text = text + line
		i+=1
	text = list(text)
	for i in range(len(text)):
		text[i] = ord(text[i])
	return np.asarray(text).astype(np.float32,copy=False)

def get_words():
	f = open('coopertext.txt','r')
	text = ''
	while True:
		line = f.readline()
		if not line:
			break
		line = re.sub(' +',' ',line)
		line = re.sub('\n',' ',line)
		line = re.sub(r"[^\w\s]+", "", line)
		line = line.lower()
		text = text + line
	text = list(text.split())
	return text


# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words, vocabulary_size):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

def get_data_vectors(vec_len=10,end=1):
	text= getText(end=1)
	x = []
	y = []
	for i in range((len(text)-1)//vec_len):
		in_var = text[i:vec_len+i]
		out_var = text[i+vec_len]
		x.append(in_var)
		ov = np.zeros(128)
		ov[int(out_var)] = 1
		y.append(ov)
	np.asarray(x).astype(np.float32,copy=False)
	np.asarray(y).astype(np.float32,copy=False)
	return[x,y]
		
