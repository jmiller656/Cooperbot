'''
This file is intended to be the script
that this project will use to gather the
anderson cooper dataset that will be used to 
train cooperbot
'''
import requests
import re
from bs4 import BeautifulSoup as bs
f = open("coopertext.txt","w")
for year in range(4,17):
	for month in range(1,13):
		for day in range(1,32):
			for time in range(1,100):
				y = str(year)
				m = str(month)
				d = str(day)
				t = str(time)
				if year <10:
					y = "0" + y
				if month <10:
					m = "0" + m
				if day <10:
					d = "0" + d
				if time <10:
					t = "0" + t
				try:		
					webpage = requests.get("http://transcripts.cnn.com/TRANSCRIPTS/" + y +m + "/" + d +"/acd." + t +".html")
				except:
					continue
				if webpage.status_code == 404:
					print "whoops,404 on " + "http://transcripts.cnn.com/TRANSCRIPTS/" + y +m + "/" + d +"/acd." + t +".html"
					break
				parser = bs(webpage.content,"html.parser")
				data = parser.find_all("p",{"class":"cnnBodyText"})
				for paragraph in data:
					try:
						text = paragraph.text
						m = re.findall('[A-Za-z %s(),-]+:',text)
						l = re.split('[A-Za-z %s(),-]+:',text)
						i = 0
						while i< len(l):
							if l[i].isspace() or l[i] == '':
								l.remove(l[i])
							else:
								i+=1
						for i in range(len(m)):
							if "COOPER" in m[i]:
								f.write(l[i]+"\n")
								print l[i]
								pass
					except Exception as e:
						print e
						print "failed to get text"
						pass
f.close()
