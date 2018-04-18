import nltk ,time
from nltk.corpus import wordnet
from collections import defaultdict
#from PyDictionary import PyDictionary
#dictionary=PyDictionary()
from nltk import pos_tag, word_tokenize
music_list=['acoustic','atonal','bass','bluesy','classical','discordant','flat','funky','harmonic','harmonious','indie','instrumental','jazzy','leftfield','lo-fi','melodic','musical','musically','off-key','orchestral','philharmonic','pizzicato','playable','polyphonic','progressive','sad','session','sharp','solo','staccato','tonal','unaccompanied','unplugged','up-tempo','vocal']



"""
apps=[]

adj_list=defaultdict(int)
k=[]

print len(music_list)

while(len(music_list)!=0):
	temp=music_list.pop()
	k.append(temp)
	if(temp not in adj_list):
		adj_list[temp]=1
	synonyms=wordnet.synsets(temp)
	for word in synonyms:
		if word.pos()=='a':
			for name in word.lemma_names():
				if(adj_list[name.encode()]!=1):
					adj_list[name.encode()]=1
					music_list.append(name.encode())

print k
print len(k)
"""
