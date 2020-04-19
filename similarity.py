import gensim
import re

def mostSimilarDocument(inDoc):
    inDoc = re.sub(r'[^\w\s]', '', inDoc)
    inDoc = re.sub(r'\S*@\S*\s?', '', inDoc, flags=re.MULTILINE)  # remove email
    inDoc = re.sub(r'http\S+', '', inDoc, flags=re.MULTILINE)  # remove web addresses
    inDoc = re.sub("\'", "", inDoc)  # remove single quotes
    inDoc = inDoc.lower()
    inDoc = inDoc.split()
    model = gensim.models.Doc2Vec.load("doc2vec.model")
    inferred_vector = model.infer_vector(inDoc)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    return [int(sims[1][0]),int(sims[2][0]),int(sims[3][0])]

