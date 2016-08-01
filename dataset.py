import re
import numpy as np
import collections


def getText():
	f = open('coopertext.txt','r')
	text = ''
	while True:
		line = f.readline()
		if not line:
			break
		line = re.sub(' +',' ',line)
		line = re.sub('\n',' ',line)
		text = text + line
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
