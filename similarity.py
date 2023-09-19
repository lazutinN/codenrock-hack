import re
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


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


def text_similarity(text1, text2):
    tokens1 = preprocess(text1)
    tokens2 = preprocess(text2)

    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform([tokens1, tokens2])
    similarity = cosine_similarity(vector)[0][1]

    return similarity
