
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset,DatasetDict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

def load_split_data(jsonl_file,train_test_split):
 
    dataset = load_dataset('json', data_files= jsonl_file)['train']
    dataset = dataset.remove_columns(["id","thread_id","thread","keyword","starting_index","span_length"])
    dataset = dataset.rename_column("TOXIC", "label")

    train_test = dataset.train_test_split(test_size=train_test_split)
    train_test_dataset = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']
    })

    full_datasets = train_test_dataset

    return full_datasets

test_train = load_split_data("./FINETUNE_DATASET.jsonl", 0.2)

train_data = test_train['train']
train_text_data = train_data['text']
labels = train_data['label'] 

# Pipeline and fitting model
""" 
Models: 
NAIVE BAYES
MultinomialNB(),¨

STOCHASTIC GRADIENT DESCENT
SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, random_state=42,
                         max_iter=5, tol=None)),

MULTI-LAYER PERCEPTRON CLASSIFIER
MLPClassifier(solver='lbfgs', hidden_layer_sizes=50,
                                max_iter=150, shuffle=True, random_state=1,
                                activation=activation)
Tokenizers:
TfidfTransformer(),

"""
text_clf = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),     
('clf', MLPClassifier(solver='lbfgs', hidden_layer_sizes=50,
                                max_iter=150, shuffle=True, random_state=1,
                                activation='logistic'))])          

text_clf.fit(train_text_data, labels)

# Evaluating model
test_data = test_train['test']
test_text_data = test_data['text']
test_labels = test_data['label']
predicted = text_clf.predict(test_text_data)
accuracy = np.mean(predicted == test_labels)

print(accuracy)
print(metrics.classification_report(test_labels, predicted))

#########################################################################
# from sklearn.dummy import DummyClassifier
# data = np.array([-1, 1, 1, 1])
# labels = np.array([0, 1, 1, 1])
# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(data, labels)
# DummyClassifier(strategy='most_frequent')
# predict = dummy_clf.predict(data)
# score = dummy_clf.score(data, labels)
# print(predict)
# print(score)

""" 
“most_frequent”: the predict method always returns the most frequent class label in the observed y argument passed to fit. The predict_proba method returns the matching one-hot encoded vector.

“prior”: the predict method always returns the most frequent class label in the observed y argument passed to fit (like “most_frequent”). predict_proba always returns the empirical class distribution of y also known as the empirical class prior distribution.

“stratified”: the predict_proba method randomly samples one-hot vectors from a multinomial distribution parametrized by the empirical class prior probabilities. The predict method returns the class label which got probability one in the one-hot vector of predict_proba. Each sampled row of both methods is therefore independent and identically distributed.

“uniform”: generates predictions uniformly at random from the list of unique classes observed in y, i.e. each class has equal probability.

“constant”: always predicts a constant label that is provided by the user. This is useful for metrics that evaluate a non-majority class.
"""