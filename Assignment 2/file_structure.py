import os

if not os.path.exists("./data/output/models"):
    print ("Not exists. Creating!!")
    os.mkdirs('./data/output/models')

if not os.path.exists("./data/output/embeddings"):
    print ("Not exists. Creating!!")
    os.mkdirs('./data/output/embeddings')