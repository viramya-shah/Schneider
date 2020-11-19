import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from my_utils import read_data, save_csv
from wordcloud import WordCloud, STOPWORDS 


if __name__ == '__main__':
    FILE_PATH = './data/output/'
    FILE_NAME = 'clean_data.csv'

    # reading clean data
    clean_df = read_data(FILE_PATH, FILE_NAME)

    # wordcloud
    wordcloud = WordCloud(
        width = 800, 
        height = 400, 
        background_color ='white', 
        stopwords = set(STOPWORDS), 
        min_font_size = 5).generate(''.join([i for i in clean_df.description]))

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.savefig("./data/output/wordcloud.png")
    plt.show() 

    # getting the embedding
    embedding_list = pickle.load(
        open("./data/output/embeddings/final_emb.pkl", 'rb'))
    clean_df['embedding'] = embedding_list

    # applying pca
    explained_variance_ratio = []
    for i in range(1, 50):
        pca = PCA(n_components=i)
        pca.fit(embedding_list)
        explained_variance_ratio.append(sum(pca.explained_variance_ratio_))
        if i % 10 == 0:
            print("{} components explain {:.3f} variance".format(
                i, sum(pca.explained_variance_ratio_)))
    plt.plot(explained_variance_ratio)
    plt.savefig("./data/output/explained_variance.png")
    plt.show()

    # applying tsne
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12,
        n_iter=1000,
        metric='euclidean'
    )
    X_tsne = tsne.fit_transform(embedding_list)

    # applying KMeans using tSNE embeddings
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        max_iter=500,
        n_init=10,
        verbose=1
    ).fit(X_tsne)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_, alpha=0.2)
    plt.title('Clusters')
    plt.savefig("./data/output/scatter_plot.png")
    plt.show()

    clean_df['labels'] = kmeans.labels_
    save_csv(clean_df, "./data/output/", "post_cluster.csv")
