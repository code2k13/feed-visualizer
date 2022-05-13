#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import shutil

import feedparser
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generates cool visualization from Atom/RSS feeds !')
parser.add_argument('-c', '--configuration', required=True,
                    help='location of configuration file.')
args = parser.parse_args()

with open(args.configuration, 'r') as config_file:
    config = json.load(config_file)
 
semantic_encoder_model = SentenceTransformer(config["pretrained_model"])

def get_all_entries(path):    
    all_entries = {}
    files = glob.glob(path+"/**/**/*.*",recursive=True)
    for file in tqdm(files, desc='Reading posts from files'):
         
        feed = feedparser.parse(file)
        for entry in feed['entries']:        
            if 'summary' in entry:
                all_entries[entry['link']] = [
                    entry['title'], entry['title'] + " " + entry['summary']]
            elif 'content' in entry:
                all_entries[entry['link']] = [
                    entry['title'], entry['title'] + " " + entry['content'][0]['value']]
    return all_entries


def generate_text_for_entry(raw_text,entry_counts):
    output = []
    raw_text = raw_text.replace("\n", " ")
    soup = BeautifulSoup(raw_text, features="html.parser")
    output.append(soup.text)
    for link in BeautifulSoup(raw_text, parse_only=SoupStrainer('a'), features="html.parser"):
        if link.has_attr('href'):
            url = link['href']
            if url in entry_counts:
                entry_counts[url] = entry_counts[url] + 1
            else:
                entry_counts[url] = 0

    return ' ' .join(output)


def generate_embeddings(entries,entry_counts):
    sentences = [generate_text_for_entry(
        entries[a][1][0:config["text_max_length"]],entry_counts) for a in entries]
    print('Generating embeddings ...')
    embeddings = semantic_encoder_model.encode(sentences)
    print('Generating embeddings ... Done !')
    index = 0
    for uri in entries:
        entries[uri].append(embeddings[index])
        index = index+1
    return entries


def get_coordinates(entries):
    X = [entries[e][-1] for e in entries]
    X = np.array(X)
    tsne = TSNE(n_iter=config["tsne_iter"],init='pca',learning_rate='auto')
    clustering_model = AgglomerativeClustering(distance_threshold=config["clust_dist_threshold"], n_clusters=None)
    clusters = clustering_model.fit_predict(tsne.fit_transform(X))
    return [x[0] for x in tsne.fit_transform(X)], [x[1] for x in tsne.fit_transform(X)], clusters


def main():
    all_entries = get_all_entries(config["input_directory"])
    entry_counts = {}
    entry_texts = []
    disinct_entries = {}
    for k in all_entries.keys():
        if all_entries[k][0] not in entry_texts:
            disinct_entries[k] = all_entries[k]
            entry_texts.append(all_entries[k][0])

    all_entries = disinct_entries
    entries = generate_embeddings(all_entries,entry_counts)
    print('Creating clusters ...')
    x, y, cluster_info = get_coordinates(entries)
    print('Creating clusters ... Done !')
    labels = [entries[k][0] for k in entries]


    counts = [entry_counts[k] if k in entry_counts else 0 for k in entries]

    df = pd.DataFrame({'x': x, 'y': y, 'label': labels,
                    'count': counts, 'url': entries.keys(), 'cluster': cluster_info})


    if not os.path.exists(config["output_directory"]):
        os.makedirs(config["output_directory"])
    df.to_csv(config["output_directory"]+"/data.csv")

    shutil.copy('visualization.html', config["output_directory"])
    print('Vizualization generation is complete !!')

if __name__ == "__main__":
    main()
