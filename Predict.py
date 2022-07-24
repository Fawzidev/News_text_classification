import pickle
import string
import nltk
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

class predict():
    docs_news=''
    
    def predict(docs_new):
        category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
        docs_new = [docs_new]

        # Loading the model
        loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
        loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
        loaded_model = pickle.load(open("svm.pkl","rb"))


        stemmer = SnowballStemmer('english')
        docs_new=[' '.join([stemmer.stem(word) for word in nltk.tokenize.word_tokenize(text) if word not in string.punctuation])
                  for text in docs_new]

        X_new_counts = loaded_vec.transform(docs_new)
        X_new_tfidf = loaded_tfidf.transform(X_new_counts)
        predicted = loaded_model.predict(X_new_tfidf)

        print(category_list[predicted[0]])
        return category_list[predicted[0]]