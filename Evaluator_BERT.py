import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import optparse
import logging
import codecs
import heapq
import pickle



NUM_SEMANTIC_CLASSES = 6
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def cosine_vector_similarity(vec_a, vec_b):
   sim = np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)* np.linalg.norm(vec_b))
   return sim

def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_sentence_embedding(sentence):
    encoding = tokenizer.batch_encode_plus(
        [sentence],				 # List of input texts
        padding=True,			 # Pad to the maximum sequence length
        truncation=True,		 # Truncate to the maximum sequence length if necessary
        return_tensors='pt',	 # Return PyTorch tensors
        add_special_tokens=True # Add special tokens CLS and SEP
    )

    input_ids = encoding['input_ids'] # Token IDs
    attention_mask = encoding['attention_mask'] # Attention mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state # This contains the embeddings

    # Compute the average of word embeddings to get the sentence embedding
    sentence_embedding = word_embeddings.mean(dim=1) # Average pooling alo

    return sentence_embedding

def create_vocab(word_freq_path):
    emb_vocab = dict()

    file1 = open(word_freq_path, 'r')
    Lines = file1.readlines()

    for line in Lines:
        line = line.strip()  
        elements = line.split()
        word =  elements[0]
        
        emb = tuple(get_word_embedding(word))
        emb_vocab[emb] = word

    file1.close()

    return emb_vocab

def get_analogy(emb_vocab, w1, w2, w3, topN=1):
    # Encode word vectors for w1, w2, and w3
    w1_vector = get_word_embedding(w1)
    w2_vector = get_word_embedding(w2)
    w3_vector = get_word_embedding(w3)

    # Compute analogy vector
    analogy_vector = w2_vector - w1_vector + w3_vector

    # Get embeddings from vocabulary
    emb_list = emb_vocab.keys()

    # Init similarities list
    similarities = []

    # Calculate similarities
    for emb in emb_list:
        if np.array_equal(emb,w1_vector) or np.array_equal(emb,w2_vector) or np.array_equal(emb, w3_vector):
            sim = cosine_vector_similarity(analogy_vector, emb)
            similarities.append((-sim, emb))  # Use negative similarity for a max-heap

    # Creat heap to find topN
    topn_nearest = heapq.nlargest(topN, similarities)

    # Convert heap to list
    nearest_embeddings = [emb for _, emb in topn_nearest]

    # Get words by embeddings
    nearest_words = [emb_vocab[emb] for emb  in nearest_embeddings]
    return nearest_words

def evaluate_file(filePath, topN, outputFile,emb_vocab):

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

                    # list = most_similar_to_vec(result_vector(tokens[0], tokens[1], tokens[2], model),model,topN,tokens[:-1])
                    list = get_analogy(emb_vocab,tokens[0],tokens[1],tokens[2],topN)
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
    if count != 0:
        semanticAcc = avgVal / count
    else:
        semanticAcc = 0

    avgVal = 0.0
    count= 0.0
    for val in listAccSynt:
        avgVal = avgVal +val
        count = count + 1.0
    if count != 0:
        syntacticAcc = avgVal / count
    else:
        syntacticAcc = 0


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

if __name__ == "__main__":
    parser = optparse.OptionParser(usage="%prog [OPTIONS]")
    parser.add_option('-m', '--model', default='./models/vectors_cz_cbow_dim300_w10_phrase.txt',
                      help='Give a path with the name of a model to load (default name= vector.txt)')
    parser.add_option('-c', '--corpus', default='./corpus/diacritics/czech_emb_corpus.txt',
                      help='Give a name of corpus to analyze  (default: ./corpus/diacritics/czech_emb_corpus.txt)')
    parser.add_option('-v', '--vocab', default='./vocabulary/diacritics/cs_50k.txt',
                      help='Give a vocabulary to use for searching analogies(default: ./vocabulary/diacritics/cs_50k.txt)')
    parser.add_option('-t', '--topn', default='1',
                      help='TOP N similar words')   
    options, args = parser.parse_args()



    # Parameters
    model_path = ''


    print("Setting the model!")
    global model
    global tokenizer
    if model_path == '':
        # Load BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
        model = BertModel.from_pretrained('google-bert/bert-base-multilingual-cased')
    else:
        pass

    print("Generating the vocabulary!")
    
    # emb_vocab = create_vocab(options.vocab)
    # Save
    # with open('cache/mBERT-diacritics_cs_50k.p', 'wb') as fp:
    #     pickle.dump(emb_vocab, fp, protocol=pickle.HIGHEST_PROTOCOL)


    # Load
    with open('cache/mBERT-diacritics_cs_50k.p', 'rb') as fp:
        emb_vocab = pickle.load(fp)


    print("Start evaluation!")
    evaluate_file(options.corpus,int(options.topn), options.model, emb_vocab)
    # print(get_analogy("boy", "prince", "girl"))

