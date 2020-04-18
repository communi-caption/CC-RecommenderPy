import gensim
import codecs
import re

def setTrainingData(trainDoc):
    with codecs.open(trainDoc, 'r', encoding='utf8') as w:
        Lines = w.readlines()
    with codecs.open("clearTrain.txt", 'w', encoding='utf8') as f:
        for line in Lines:
            line = re.sub(r'[^\w\s]', '', line)
            line = re.sub(r'\S*@\S*\s?', '', line, flags=re.MULTILINE)  # remove email
            line = re.sub(r'http\S+', '', line, flags=re.MULTILINE)  # remove web addresses
            line = re.sub("\'", "", line)  # remove single quotes
            line = line.lower()
            f.write(line)
    f.close()
    train_corpus = list(read_corpus("clearTrain.txt"))
    getDoc2VecModel(train_corpus)

def read_corpus(fname, tokens_only=False):
    count = -1
    with codecs.open(fname, 'r', encoding='utf8') as f:
        Lines = f.readlines()
    for line in Lines:
        count += 1
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [count])

def getDoc2VecModel(train_corpus):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=15, min_count=2, epochs=20)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("doc2vec.model")