# -*- coding: utf-8 -*-
__author__ = 'jirkoada'  # Original code by Svobikl. Fixed some errors related to gensim updates.

# accepts both binary fasttext and plain-text models, as well as compressed fasttext models created via compress-fasttext package

from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import load_facebook_model
from gensim import corpora, models, similarities, matutils
import re
import os
import logging
import optparse
import numpy as np
import operator
import codecs
from fnmatch import fnmatch


NUM_SEMANTIC_CLASSES = 6
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def cosine_vector_similarity(vec_a, vec_b):

   sim = np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)* np.linalg.norm(vec_b))
   return sim

def result_vector(st1,st2,st3, model):
    vec1 = model.get_vector(st1, norm=True)
    vec2 = model.get_vector(st2, norm=True)
    vec3 = model.get_vector(st3, norm=True)

    sub_vec = list(map(operator.sub, vec2,vec1))
    result_vec = list(map(operator.add, sub_vec, vec3))

    return result_vec
def most_similar_to_vec(vector,model,topn, list_words):
    dists = np.dot(model.get_normed_vectors(), vector)

    best = matutils.argsort(dists, topn=topn + len(list_words), reverse=True)
    # ignore (don't return) words from the input
    result = [(model.index_to_key[sim], float(dists[sim])) for sim in best if model.index_to_key[sim] not in list_words]

    return result[:topn]
def evaluate_file(filePath, topN, outputFile):

    accuracy = 0.0
    accuracyCosMul = 0.0
    accuracyAll = 0.0
    accuracyAllCosMul = 0.0
    classItemsCount = 0
    notSeenCounter = 0
    questionsCount =0
    classNumb = 0

    listAccSemantic = []
    listAccSynt= []
    fw = codecs.open(outputFile[:-4]+".res"+str(topN)+".txt", 'w','utf-8' )
    prevCategory = ": Antonyms-nouns"
    fwerr = codecs.open(outputFile[:-4]+"err.log", 'w', 'utf-8')

    listErr= []

    with codecs.open(filePath, 'r','utf-8') as f:
        for line in f:

            if (line.strip()[0]==':'):

                if classItemsCount!=0:

                    currAcc= (accuracy/classItemsCount)*100.0
                    currAccCosMul= (accuracyCosMul/classItemsCount)*100.0
                    if classNumb< NUM_SEMANTIC_CLASSES:
                        listAccSemantic.append(currAcc)
                    else :
                        listAccSynt.append(currAcc)
                    print(prevCategory + " > accuracy TOP%d = %f (%d/%d)\n" % (topN,currAcc, accuracy,classItemsCount))
                    fw.write(prevCategory + " > accuracy TOP%d = %f (%d/%d) \n" % (topN,currAcc, accuracy,classItemsCount))
                    prevCategory = line
                    classNumb = classNumb + 1

                print (line)
                accuracy = 0.0
                accuracyCosMul = 0.0
                classItemsCount = 0
            else:
                tokens = line.lower().strip().split(" ")
                questionsCount = questionsCount + 1.0
                classItemsCount = classItemsCount + 1.0
                try:

                    list = most_similar_to_vec(result_vector(tokens[0], tokens[1], tokens[2], model),model,topN,tokens[:-1])
                    for item in list:
                        match = item[0]

                        #match = match.encode('utf-8')
                        if match == tokens[3]:
                            #print "Correct item=%s" % (item[0])
                            accuracy =accuracy + 1.0
                            accuracyAll =accuracyAll + 1.0

                except KeyError as e:
                    logging.error(e)
                    wordErr = str(e).encode("utf-8")
                    notSeenCounter = notSeenCounter + 1.0
                    listErr.append(wordErr)

    if classItemsCount!=0:
        currAcc= (accuracy/classItemsCount)*100.0
        currAccCosMul= (accuracyCosMul/classItemsCount)*100.0
        listAccSynt.append(currAcc)
        print(prevCategory + " > accuracy TOP%d = %f (%d/%d)\n" % (topN,currAcc, accuracy,classItemsCount))
        fw.write(prevCategory + " > accuracy TOP%d = %f (%d/%d)\n" % (topN,currAcc, accuracy,classItemsCount))

    avgVal = 0.0
    count= 0.0
    for val in listAccSemantic:
        avgVal = avgVal +val
        count = count + 1.0
    semanticAcc = avgVal / count

    avgVal = 0.0
    count= 0.0
    for val in listAccSynt:
        avgVal = avgVal +val
        count = count + 1.0
    syntacticAcc = avgVal / count


    print ("Total accuracy TOP%d = %f \n" % (topN,(accuracyAll/questionsCount)*100.0))
    fw.write("Total accuracy TOP%d = %f \n" % (topN,(accuracyAll/questionsCount)*100.0))
    fw.write("Semantic accuracy TOP%d = %f \n" % (topN,semanticAcc))
    fw.write("Syntactic accuracy TOP%d = %f \n" % (topN,syntacticAcc))
    #print "Total accuracy CosMul TOP%d = %f" % (topN,(accuracyAllCosMul/questionsCount)*100.0)
    #fw.write("Total accuracy CosMul TOP%d = %d", (topN,(accuracyAllCosMul/questionsCount)*100.0))
    print ("Seen= %f" % (((questionsCount-notSeenCounter)/questionsCount) * 100.0))
    fw.write("Seen= %f"% (((questionsCount-notSeenCounter)/questionsCount) * 100.0))

    for word in np.unique(listErr):
        fwerr.write(str(word)+"\n")

    fw.close()
    fwerr.close()

    return 0


if __name__ == '__main__':

    parser = optparse.OptionParser(usage="%prog [OPTIONS]")
    parser.add_option('-m', '--model', default='./models/vectors_cz_cbow_dim300_w10_phrase.txt',
                      help='Give a path with the name of a model to load (default name= vector.txt)')
    parser.add_option('-c', '--corpus', default='./corpus/diacritics/czech_emb_corpus.txt',
                      help='Give a name of corpus to analyze  (default: ./corpus/diacritics/czech_emb_corpus.txt)')
    parser.add_option('-t', '--topn', default='1',
                      help='TOP N similar words')
    parser.add_option('--compressed', default=False, action="store_true",
                      help='Indicate the model was compressed with compress-fasttext')
    options, args = parser.parse_args()


    if options.model[-3:] == "txt" or options.model[-3:] == "vec":
        model = models.KeyedVectors.load_word2vec_format(options.model, binary=False)
    else:
        if options.compressed:
            import compress_fasttext
            model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(options.model)
        else:
            model = load_facebook_model(options.model)
            model = model.wv
    model.fill_norms()
    evaluate_file(options.corpus,int(options.topn), options.model)
