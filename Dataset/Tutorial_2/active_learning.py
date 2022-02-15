#from load_data import * 
import pandas as pd
import matplotlib as plt
from sklearn.naive_bayes import MultinomialNB
from modAL.models import ActiveLearner
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import NotFittedError
import rubrix as rb

'''
* Annotate samples

* Teach the active learner

* Plot the improvement

- Very repetitive code now , also creating several datasets in rubrix.
- Could maybe add a userinput for when to continue with the loop ? After you have annotated all the data in rubrix. 

'''

# ---- 1. load data ---- #
train_df = pd.read_csv ("data/active_learning/train.csv")
test_df = pd.read_csv ("data/active_learning/test.csv")


# ---- 2. Defining our classifier and Active Learner ---- #
classifier = MultinomialNB()

learner = ActiveLearner(
    estimator=classifier,
)

vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(train_df.CONTENT)
X_test = vectorizer.transform(test_df.CONTENT)


# ---- 3. Active Learning loop ---- #

n_instances = 10
accuracies = []

# query examples from our training pool with the most uncertain prediction
query_idx, query_inst = learner.query(X_train, n_instances=n_instances)

# -- Initialization -- #

# # get predictions for the queried examples
# try:
#     probabilities = learner.predict_proba(X_train[query_idx])
# # For the very first query we do not have any predictions
# except NotFittedError:
#     probabilities = [[0.5, 0.5]]*n_instances

# # Build the Rubrix records
# records = [
#     rb.TextClassificationRecord(
#         id=idx,
#         inputs=train_df.CONTENT.iloc[idx],
#         prediction=list(zip(["HAM", "SPAM"], probs)),
#         prediction_agent="MultinomialNB",
#     )
#     for idx, probs in zip(query_idx, probabilities)
# ]

# Log the records
# rb.log(records, name="active_learning_tutorial1")


# # -- Iteration 1 -- #

# records_df = rb.load("active_learning_tutorial1", ids=query_idx.tolist())

# # check if all examples were annotated
# if any(records_df.annotation.isna()):
#     raise UserWarning("Please annotate first all your samples before teaching the model")

# # train the classifier with the newly annotated examples
# y_train = records_df.annotation.map(lambda x: int(x == "SPAM"))

# learner.teach(X=X_train[query_idx], y=y_train.to_list())

# # # Keep track of our improvement
# accuracies.append(learner.score(X=X_test, y=test_df.CLASS))

# records = [
#     rb.TextClassificationRecord(
#         id=idx,
#         inputs=train_df.CONTENT.iloc[idx],
#         prediction=list(zip(["HAM", "SPAM"], probs)),
#         prediction_agent="MultinomialNB",
#     )
#     for idx, probs in zip(query_idx, probabilities)
# ]

# # Log the records
# #rb.log(records, name="active_learning_tutorial2")

# # # -- Iteration 2  -- #

# records_df = rb.load("active_learning_tutorial2", ids=query_idx.tolist())

# # check if all examples were annotated
# if any(records_df.annotation.isna()):
#     raise UserWarning("Please annotate first all your samples before teaching the model")

# # train the classifier with the newly annotated examples
# y_train = records_df.annotation.map(lambda x: int(x == "SPAM"))

# learner.teach(X=X_train[query_idx], y=y_train.to_list())

# # # Keep track of our improvement
# accuracies.append(learner.score(X=X_test, y=test_df.CLASS))

# records = [
#     rb.TextClassificationRecord(
#         id=idx,
#         inputs=train_df.CONTENT.iloc[idx],
#         prediction=list(zip(["HAM", "SPAM"], probs)),
#         prediction_agent="MultinomialNB",
#     )
#     for idx, probs in zip(query_idx, probabilities)
# ]

# # Log the records
# #rb.log(records, name="active_learning_tutorial3")

# records_df = rb.load("active_learning_tutorial3", ids=query_idx.tolist())

# # check if all examples were annotated
# if any(records_df.annotation.isna()):
#     raise UserWarning("Please annotate first all your samples before teaching the model")

# # train the classifier with the newly annotated examples
# y_train = records_df.annotation.map(lambda x: int(x == "SPAM"))

# learner.teach(X=X_train[query_idx], y=y_train.to_list())

# # # Keep track of our improvement
# accuracies.append(learner.score(X=X_test, y=test_df.CLASS))

# records = [
#     rb.TextClassificationRecord(
#         id=idx,
#         inputs=train_df.CONTENT.iloc[idx],
#         prediction=list(zip(["HAM", "SPAM"], probs)),
#         prediction_agent="MultinomialNB",
#     )
#     for idx, probs in zip(query_idx, probabilities)
# ]

# # Log the records
# rb.log(records, name="active_learning_tutorial4")

# print(accuracies)

##################################
# Example of loop ? 
# random_samples = 50
# # max uncertainty strategy
# accuracies_max = []
# for i in range(random_samples):
#     train_rnd_df = train_df#.sample(frac=1)
#     test_rnd_df = test_df#.sample(frac=1)
#     X_rnd_train = vectorizer.transform(train_rnd_df.CONTENT)
#     X_rnd_test = vectorizer.transform(test_rnd_df.CONTENT)

#     accuracies, learner = [], ActiveLearner(estimator=MultinomialNB())

#     for i in range(n_iterations):
#         query_idx, _ = learner.query(X_rnd_train, n_instances=n_instances)
#         learner.teach(X=X_rnd_train[query_idx], y=train_rnd_df.CLASS.iloc[query_idx].to_list())
#         accuracies.append(learner.score(X=X_rnd_test, y=test_rnd_df.CLASS))
#     accuracies_max.append(accuracies)