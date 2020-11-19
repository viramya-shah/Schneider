import os

if not os.path.exists("./data/output/models"):
    print ("Not exists. Creating Models!!")
    os.mkdir('./data/output/models')

if not os.path.exists("./data/output/embeddings"):
    print ("Not exists. Creating Embedding!!")
    os.mkdir('./data/output/embeddings')