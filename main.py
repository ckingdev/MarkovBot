__author__ = "Cameron Palone"
__copyright__ = "Copyright 2015, Cameron Palone"
__credits__ = ["Cameron Palone"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Cameron Palone"
__email__ = "cam@cpalone.me"
__status__ = "Prototype"

from nltk.corpus import webtext

import praw

import models

m = models.TrigramBackoffLM()
# for file_id in webtext.fileids():
#     m.update([webtext.raw(file_id)])
# print(m.generate())

# r = praw.Reddit("MarkovBot by /u/LackingIsntEmpty")
# user = r.get_redditor("greenduch")
# m.update([post.body for post in user.get_comments(limit=10000)])
# m.save("greenduch.pickle")
# print(m.generate())
# print(m.generate())
# print(m.generate())
# print(m.generate())
# print(m.generate())
# print(m.generate())
# print(m.generate())
# print(m.generate())
with open("data/tomsawyer.txt") as f:
    m.update([f.read()])
print(m.n_unigram)
print(m.n_bigram)
print(m.n_trigram)
print("Completed training.")
print(m.generate())
print(m.generate())
print(m.generate())
print(m.generate())
print(m.generate())
print(m.generate())
print(m.generate())
print(m.generate())
