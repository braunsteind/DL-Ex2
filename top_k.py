import utils1 as ut1
import numpy as np
from numpy import linalg as la


# TODO: delete
def main():
    print "Top 5 most similar to dog: " + str(most_similar("dog", 5))
    print "Top 5 most similar to England: " + str(most_similar("england", 5))
    print "Top 5 most similar to John: " + str(most_similar("john", 5))
    print "Top 5 most similar to explode: " + str(most_similar("explode", 5))
    print "Top 5 most similar to office: " + str(most_similar("office", 5))


def most_similar(word, k):
    """
    most_similar function.
    returns k most similar words to the input word.
    :param word: the requested word
    :param k: number of close words we want to find.
    :return: list of the k most similar words.
    """

    # TODO: change ut1 to helper
    # get the word
    words = ut1.WORD_EMBEDDINGS_DICT
    word = words[word]
    # set the distance array
    distances = []
    # loop over words
    for i in words:
        # calc the distance between word and words[i]
        dist = cosine_distance(word, words[i])
        # add the distance to array
        distances.append([i, dist])

    # sort the array
    distances = sorted(distances, key=get_distance)
    # get top k
    top_k = sorted(distances, key=get_distance, reverse=True)[1:k + 1]
    top_k = [item[0] for item in top_k]
    return top_k


def cosine_distance(u, v):
    """
    calculates the distance between u and v.
    :param u: Vector
    :param v: Vector
    :return: The distance between u and v.
    """
    return (np.dot(u, v)) / (np.max([float(la.norm(u, 2) * la.norm(v, 2)), 1e-8]))


def get_distance(word_dist):
    return word_dist[1]


if __name__ == "__main__":
    main()
