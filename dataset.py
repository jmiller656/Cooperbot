import re
import numpy as np
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
