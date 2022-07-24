import pandas
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
import nltk
import string
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

class Model():
    def __init__(self):
        # Load data
        self.training_data = pandas.read_csv('train_data.csv')

    def Classifier(self):
        # Counting vector
        count_vect = CountVectorizer(ngram_range=(1,3),stop_words=frozenset(stopwords.words('english')[0:-1]))

        stemmer = SnowballStemmer('english')
        corpus=[' '.join([stemmer.stem(word) for word in nltk.tokenize.word_tokenize(text) if word not in string.punctuation])
                  for text in self.training_data.data]

        X_train_counts = count_vect.fit_transform(corpus)

        pickle.dump(count_vect.vocabulary_, open("count_vector.pkl", "wb"))

        # Weights assignement
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        pickle.dump(tfidf_transformer, open("tfidf.pkl", "wb"))

        # Training
        svm_classification = svm.LinearSVC(class_weight='balanced')#stagnation a partir de 10 iters
        X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, self.training_data.flag, test_size=0.20)

        svm_classification.fit(X_train_tfidf, self.training_data.flag)
        pickle.dump(svm_classification, open("svm.pkl", "wb"))
        predicted = svm_classification.predict(X_test)
        result_svm = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
        result_svm.to_csv('res_svm.csv', sep = ',')

        count = ['sport', 'world', "us", "business", "health", "entertainment", "sci_tech"]

        Eval = precision_recall_fscore_support(predicted, y_test, average='macro')

        print(classification_report(y_test, predicted, target_names=count))
        print('Accuracy =',accuracy_score(predicted,y_test,normalize=True))
        print('Precision = ',Eval[0])
        print('Rappel = ', Eval[1])
        print('f-mesure = ', Eval[2])
        roc_curve(predicted,y_test,pos_label=7)