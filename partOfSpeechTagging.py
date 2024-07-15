#NLTK POS Tagger

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

text1 = word_tokenize("And now for something completely different")
# print(nltk.pos_tag(text1))

text2 = word_tokenize("They refuse to permit us to obtain the refused permit")
# print(nltk.pos_tag(text2))

nltk.help.upenn_tagset()

# # text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
# word_list = ['woman', 'bought', 'over', 'the']
# for w in word_list:
#     # print('\nwords in text similar to ' + w + " are: ")
#     text.similar(w)
    
# Representing Tagged Tokens
tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)

# Reading tagged corpora
print(nltk.corpus.brown.tagged_words(tagset='universal'))

