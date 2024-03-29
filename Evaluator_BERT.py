import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import optparse
import logging
import codecs
import pickle
import os

NUM_SEMANTIC_CLASSES = 6
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
    sentence_embedding = word_embeddings.mean(dim=1) # Average the tokens embeddings

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
    """
    Finds top N nearest words using NumPy and PyTorch (if available).
    Utilizes GPU acceleration if a CUDA device is present.

    Args:
        emb_vocab: Dictionary mapping embeddings to their words.
        analogy_vector: Embedding vector for the analogy word.
        w1, w2, w3: word to generate analogy.
        topN: Number of nearest neighbors to find.

    Returns:
        nearest_words: List of top N nearest words based on cosine similarity.
    """
    # Encode word vectors for w1, w2, and w3
    w1_vector = get_word_embedding(w1)
    w2_vector = get_word_embedding(w2)
    w3_vector = get_word_embedding(w3)

    # Compute analogy vector
    analogy_vector = w2_vector - w1_vector + w3_vector

    # Convert embeddings to NumPy array
    emb_array = np.array(list(emb_vocab.keys()))

    # Check for CUDA availability
    if torch.cuda.is_available():
        w1_tensor = torch.tensor(w1_vector).cuda()
        w2_tensor = torch.tensor(w2_vector).cuda()
        w3_tensor = torch.tensor(w3_vector).cuda()
        analogy_vector = torch.tensor(analogy_vector).cuda()
        emb_tensor = torch.tensor(emb_array).cuda()

        # PyTorch vectorized comparison and similarity (GPU)
        mask = torch.all(emb_tensor != w1_tensor, dim=1) & torch.all(emb_tensor != w2_tensor, dim=1) & torch.all(emb_tensor != w3_tensor, dim=1)
        similarities = torch.cosine_similarity(emb_tensor[mask], analogy_vector, dim=1)
        topn_indices = torch.topk(similarities, topN).indices
        nearest_embeddings = emb_tensor[mask][topn_indices].cpu().numpy()  # Transfer back to CPU
    else:
        # Fallback to NumPy (CPU)
        mask = ~(np.linalg.norm(emb_array - w1_vector, axis=1) == 0) & \
            ~(np.linalg.norm(emb_array - w2_vector, axis=1) == 0) & \
            ~(np.linalg.norm(emb_array - w3_vector, axis=1) == 0)

        # Vectorized cosine similarity calculation (NumPy)
        similarities = -np.dot(emb_array[mask], analogy_vector) / (
            np.linalg.norm(emb_array[mask], axis=1) * np.linalg.norm(analogy_vector)
        )

        # Get topN indices using NumPy's argpartition
        topn_indices = np.argpartition(similarities, -topN)[-topN:]
        nearest_embeddings = emb_array[mask][topn_indices]

    # Reconstruct words from nearest embeddings
    nearest_words = [emb_vocab[tuple(emb)] for emb in nearest_embeddings]
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
    fw = codecs.open(outputFile+".res"+str(topN)+".txt", 'w','utf-8' )
    prevCategory = ": Antonyms-nouns"
    fwerr = codecs.open(outputFile+"err.log", 'w', 'utf-8')

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
    parser.add_option('-p', '--pretrained_model', default='',
                      help='Give a name of a pretrained model to load')
    parser.add_option('-c', '--corpus', default='./corpus/diacritics/czech_emb_corpus.txt',
                      help='Give a name of corpus to analyze  (default: ./corpus/diacritics/czech_emb_corpus.txt)')
    parser.add_option('-v', '--vocab', default='./vocabulary/diacritics/cs_50k.txt',
                      help='Give a vocabulary to use for searching analogies(default: ./vocabulary/diacritics/cs_50k.txt)')
    parser.add_option('-t', '--topn', default='1',
                      help='TOP N similar words')
    parser.add_option('-o', '--output', default='./output/',
                      help='Path for saving results of models. (default: ./output/ )')
    parser.add_option( '--cached_vocab', default='',
                      help='Path of created vocabulary. (disabled by default )')
    options, args = parser.parse_args()


    print("Setting the model!")
    global model
    global tokenizer

    if options.pretrained_model:
        print(f"Using pretrained model {options.pretrained_model}!")
        
        tokenizer = BertTokenizer.from_pretrained(options.pretrained_model)
        model = BertModel.from_pretrained(options.pretrained_model)
    else:
        print(f"Using local model!")
        # TODO: load local model
        pass

    
    if options.pretrained_model:
        output_path = os.path.join(options.output,options.pretrained_model.split("/")[-1].split("\\")[-1]) 
    else:
        output_path = os.path.join(options.output, options.model.split("/")[-1].split("\\")[-1])

    
    # Create/Load vocabulary
    if options.cached_vocab:
        print("Load vocabulary from cache!")

        with open(options.cached_vocab, 'rb') as fp:
            emb_vocab = pickle.load(fp)
    else:
        print("Generating the vocabulary!")

        emb_vocab = create_vocab(options.vocab)

        # Save
        vocab_name = options.vocab.split("/")[-1].split("\\")[-1]
        vocab_path = os.path.join(options.output,'cache/',vocab_name + '.p')
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)  # Create parent directories

        with open(vocab_path, 'wb') as fp:
            pickle.dump(emb_vocab, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Print hardware info
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = "CUDA"
    else:
        print("CUDA is not available :(")
        device = "CPU"
    print(f"Computing on {device}.")


    print("Start evaluation!")
    evaluate_file(options.corpus,int(options.topn), output_path, emb_vocab)

    # emb = get_word_embedding('nejmodernější')
    # print(emb.shape)


