import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import scipy


df = pd.read_csv("D:\Year 3\Text analytics and sentiment analysis\ASSIGNMENT\Assignment Data\Fake_news\Fake_news_dataset.csv")
pd.set_option('display.max_colwidth', 20)

colors = ['red']

def EDA(df):
    # view top of data
    print("Data top entries:")
    print("-"*100)
    print(df.head())
    
    # view bottom of data
    print("\n\nData tail entries:")
    print("-"*100)
    print(df.tail())
    
    # rows and columns of data
    rows, columns = df.shape
    print("\nData columns:", columns)
    print("Data rows:", rows)
    
    # target variable distribution
    print("\n", df['label'].value_counts(normalize = True))
    
    # general info on data
    print('\nGeneral information about the Data')
    print("-"*100)
    print(df.info())
    
    # check missing values in data
    print('\nMissing Values in Data')
    print("-"*100)
    print(df.isnull().sum())
    
    # check for blank, and replace with null
    df = df.replace({'': None})
    
    # check if there's duplicates
    if rows > df.index.nunique():
        
        # unique values in the data
        print("\nUnique values in index column")
        print("-"*100)
        print(df.index.nunique())
        
        # number of duplicates
        print("\nDuplicated values")
        print("-"*100)
        print(f"There are {rows - df.index.nunique()} duplicates")
        
    else:
        print("\nNo duplicates found.")
        
    
    # plotting bar-graph for missing values
    missing_dataframe = pd.DataFrame(data={'Missing Values': df.isna().sum()})
    
    miss_flag = 0
    
    for row in missing_dataframe['Missing Values']:
        if row > 0:
            miss_flag = 1
            
    print("\n")
    
    if miss_flag == 1:
        (pd.DataFrame(data={'% of Missing Values':round(df.isna().
                                                    sum()/df.isna().
                                                    count()*100,2)}).
        sort_values(by = '% of Missing Values', ascending = False).
        plot(kind = 'bar', color = colors[0], figsize = (15,5), legend = False)
        )
        
        plt.title('Missing Values in percentage for Features', fontsize = 14)
        plt.xlabel('Features', fontsize = 12)
        plt.ylabel('Percentage', fontsize = 12)
        
        plt.show()
        
    else:
        print("\n No missing values")
        
    # Fake vs Real news body.length graph
    df["title_text"] = df["title"] + " " + df["text"]
    df = df[pd.notna(df["title_text"])]
    df["body_len"] = df["title_text"].apply(lambda x: len(str(x).strip()) - str(x).count(" "))
    
    bins = np.linspace(0, 200, 40)
    plt.hist(df[df["label"]== 1]["body_len"], bins, alpha=0.5, label="Fake", color="#FF5733")
    plt.hist(df[df["label"]== 0]["body_len"], bins, alpha=0.5, label="Real", color="#33FFB8")
    plt.legend(loc="upper left")
    plt.show()
    
    titles = ' '.join(title for title in df['title'])
    wordcloud = WordCloud(
        background_color='white', 
        max_words=300,
        width=800, 
        height=400,
    ).generate(titles)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    
        
EDA(df)

# # cleaning data
df = df.dropna()

df.drop(["Unnamed: 0"], axis=1, inplace=True)

print(df.head(5))

# # removing stop words and punctuation, and lemmatize the remaining text
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_lg')

# for practical purposes, sampling had to be done, since the dataset is large
sample_size = int(len(df) * 0.20)

sample_data = df.sample(n=sample_size, random_state=1)

pd.options.display.max_colwidth = 100

print(sample_data['text'])

def remove_stop_punc(text):
    doc = nlp(text)
    
    filtered_tokens = []
    
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
   
    return " ".join(filtered_tokens)

sample_data['text_new'] = sample_data['text'].apply(remove_stop_punc)

sample_data = sample_data.drop(columns = 'text')

print(sample_data['text_new'])

sample_data['text_new'] = sample_data['text_new'].str.replace('\n', ' ')

sample_data.to_csv('sample_data.csv', index=False)


    
    