text = ["London Paris London","Paris Paris London"]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

"""
['london', 'paris']
[[2 1]
 [1 2]]"""

from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)

"""[[1.  0.8]
 [0.8 1. ]]"""


# df create ....

#special column
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

#fit this similarity

# for advertisement x

#return list of students 
listofsimilar_entites = get_advertisement_similarity()

#sort them

# get first 20
