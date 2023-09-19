import pandas as pd
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from nltk.corpus import stopwords
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

#natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

nltk.download('stopwords')
nltk.download('punkt')

def datasetPreprocessing(file_path):
    df = pd.read_csv(file_path, delimiter=',', encoding='utf-8', engine='python')
    df_copy = df
    df_copy['text'] = df_copy['text'].fillna('')
    df_copy['text'] = df_copy['text'].astype(str)
    return df_copy

def preprocess(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    emoji = re.compile("["
                      u"\U0001F600-\U0001F64F"
                      u"\U0001F300-\U0001F5FF"
                      u"\U0001F680-\U0001F6FF"
                      u"\U0001F1E0-\U0001F1FF"
                      u"\U00002500-\U00002BEF"
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"
                      u"\u3030"
                      "]+", re.UNICODE)
    text = emoji.sub(r'', text)

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    tokens = [token.lemma for token in doc.tokens]
    stop_words = stopwords.words('russian')
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)
def removeDuplicates(df):
    stop_words = set(stopwords.words('russian'))
    df['text'] = df['text'].str.lower().str.split().apply(lambda x: ' '.join([word for word in x if word not in stop_words]))

    # TF-IDF
    russian_stop_words = stopwords.words('russian')
    vectorizer = TfidfVectorizer(stop_words=russian_stop_words)
    tfidf_matrix = vectorizer.fit_transform(df['text'])

    num_rows = tfidf_matrix.shape[0]
    num_batches = 100
    batch_size = num_rows // num_batches

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i != num_batches - 1 else num_rows
        batch_matrix = tfidf_matrix[start_idx:end_idx]

        # Cosine Similarity
        cosine_similarities = linear_kernel(batch_matrix, tfidf_matrix)


    # if similarity score > 0.7 => duplicate
    duplicate_pairs = []
    for i in range(len(cosine_similarities) - 1):
        for j in range(i + 1, len(cosine_similarities)):
            if cosine_similarities[i, j] > 0.7:
                duplicate_pairs.append((i, j))

    # Removing duplicates
    to_drop = sorted(set([j for i, j in duplicate_pairs]))
    df = df.drop(to_drop)
    df.to_csv('cleaned_dataset.csv', index=False, sep=';')



if __name__ == '__main__':
    df = datasetPreprocessing('dataset_1_csv.csv')
    removeDuplicates(df)