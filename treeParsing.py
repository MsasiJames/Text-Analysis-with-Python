import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from textblob import TextBlob

# Text data
text = "The big black dog barked at the white cat and chased away."

# NLTK parts of speech tagging
tokens = word_tokenize(text)
nltk_pos_tags = pos_tag(tokens)

# TextBlob parts of speech tagging
blob = TextBlob(text)
textblob_pos_tags = [(word, tag) for word, tag in blob.tags]

# Regex Tag parts of speech tagging
patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN'),                    # nouns (default)
     (r'^\d+$', 'CD'),
     (r'.*ing$', 'VBG'),               # gerunds, i.e. wondering
     (r'.*ment$', 'NN'),               # i.e. wonderment
     (r'.*ful$', 'JJ')                 # i.e. wonderful
 ]

regexp_tagger = nltk.RegexpTagger(patterns)
regexp_pos_tags = regexp_tagger.tag(tokens)

# Output
# print("NLTK POS Tags: ", nltk_pos_tags)
# print("\nTextBlob POS Tags: ", textblob_pos_tags)
# print("\nRegular Expression POS Tags: ", regexp_pos_tags)

# Parse Tree
text2 = nltk.CFG.fromstring("""
S -> NP VP
NP -> Det Adj Adj N | PP NP | Det Adj N
VP -> ConjP Conj ConjP
ConjP -> V NP | V Adv
Conj -> 'and'
V -> 'barked' | 'chased'
N -> 'dog' | 'cat'
Det -> 'The' | 'the'
Adj -> 'big' | 'black' | 'white'
Adv -> 'away'
PP -> 'at'
""")
text1 = nltk.tokenize.word_tokenize("The big black dog barked at the white cat and chased away")
print(text1)
print()
parser = nltk.ChartParser(text2)
for tree in parser.parse(text1):
   tree.draw()
   
    
