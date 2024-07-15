from nltk.tokenize import sent_tokenize, word_tokenize
from urllib.request import urlopen
from nltk.corpus import stopwords
import nltk


sampleText = "This programme is designed to provide students with knowledge and applied skills in data science, big data analytics and business intelligence. It aims to develop analytical and investigative knowledge and skills using data science tools and techniques, and to enhance data science knowledge and critical interpretation skills. Students will understand the impact of data science upon modern processes and businesses, be able to identify, and implement specific tools, practices, features and techniques to enhance the analysis of data."

sentences = sent_tokenize(sampleText)
tokens = word_tokenize(sampleText)

print("There are: ", len(sentences), " sentences in the sample text")
print("There are: ", len(tokens), " tokens in the sample text")

counter = 0
for sent in sentences:
    counter += 1
    print(counter, ".", sent, "\n")
    
counter = 0
for tokens in tokens:
    counter += 1
    print(counter, ".", tokens)
    
# reading text from the web and removing stop words

url = "http://www.gutenberg.org/files/2554/2554-0.txt"

urlText = urlopen(url).read().decode('utf8')
# print(urlText)

urlTextTokens = word_tokenize(urlText)
stopUrlTextTokens = stopwords.words("english")
filteredTokens = []

for words in urlTextTokens:
    if words not in stopUrlTextTokens:
        filteredTokens.append(words)
        
print("There are ", len(filteredTokens), " words in this text after removing stop words")
# print(filteredTokens)

f = nltk.FreqDist(filteredTokens)
print(f.most_common(20))
