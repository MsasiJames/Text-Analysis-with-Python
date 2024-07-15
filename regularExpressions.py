import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

word = 'supercalifragilisticexpialidocioussuper'

print(re.findall(r'super', word))
print(len(re.findall(r'[aeiou]', word)))

# Search
string = 'an example word:car'
match = re.search(r'word:\w\w\w', string)
if match:
    print('found')
else:
    print('no match')
    
# trying to initial vowel sequences, final vowel sequences, and all consonants
regexp = r"^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]"
def compress(word):
    pieces = re.findall(regexp, word)
    return "".join(pieces)

english_udhr = nltk.corpus.udhr.words('English-Latin1')
print(english_udhr)
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))


# extract all consonant-vowel sequences like 'ka' and 'si'

rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
print(rotokas_words)

cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
cfd = nltk.ConditionalFreqDist(cvs)

print(cfd.tabulate())

cv_word_pairs = [(cv, w) for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]

print(cv_word_pairs)

cv_index = nltk.Index(cv_word_pairs)
print(cv_index['su'])

# Finding word stems

def stem(word):
    for suffix in ['ing', 'ly', 'ious', 'ive', 'es', 's', 'ment', 'ies', 'ed']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
        
print(stem('advancement'))

    # Using regular expressions to before stem()  above
    
print(re.findall(r'^(.*?)(ing|ly|ious|ive|es|s|ment|ies|ed|ment)?$', 'Language'))

def stemTwo(word):
    regexp = r'^(.*?)(ing|ly|ious|ive|es|s|ment|ies|ed|ment)?$'
    stemTwo, suffix = re.findall(regexp, word)[0]
    
    return stemTwo

sampleText = """You probably worked out that a backslash means that
the following character is deprived of its special powers
and must literally match a specific character in the word.
Thus, while . is special, \. only matches a period. The
braced expressions, like {3,5}, specify the number of repeats
of the previous item. The pipe character indicates a choice
between the material on its left or its right. Parentheses
indicate the scope of an operator: they can be used together
with the pipe (or disjunction) symbol like this: «w(i|e|ai|oo)t»,
matching wit, wet, wait, and woot. It is instructive to see what
happens when you omit the parentheses from the last expression
above."""

tokens = nltk.tokenize.word_tokenize(sampleText)

print([stemTwo(t) for t in tokens])

text = "I am human"
tokens = re.split(' ', text)

print(tokens)

s_nums = 'one1two22three333four'
print(re.split('\d+', s_nums))

Text = input("Enter the text corpus for stemming: ")
words = nltk.tokenize.word_tokenize(Text)
print(words)

porter = nltk.PorterStemmer()
for t in words:
    print(porter.stem(t))