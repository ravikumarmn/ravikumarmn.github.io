---
title: Search NeurIPS-2022 Similar Papers
author: Ravikumar
date: 2024-05-28T01:24:33.814Z
categories:
  - Natural Language Processing
  - Document Retrieval
tags:
  - Sentence Transformer
  - S-BERT
  - NeurIPS
  - Annoy indexe
image:
  path: images/Machine Learning/Natural_Language_Processing/Search_neurips_paper/information_retrieval.png
  alt: Information Retrieval
permalink: /blogs/:title
comments: true

---


This article is about simplifing the steps of achieving **_similar document retrieval_** and deploying it in Streamlit app. Sentences are embedded with S-BERT and indexed using Annoy Index.

## **Gather data**

Here, BeautifulSoup library is used in scraping the NeurIPS-2022 papers. While thinking of scraping, the url of each paper is changing with respect to unique event id. So, It was easier to get the each paper information.

Gathered information : 
- **Title** : Research paper title
- **Abstract** : Abstract of paper
- **paper url** : link to the paper
- **paper url id** : url link id to acces the paper from id.

How data looks,

![head_scraped.png](images/Machine Learning/Natural_Language_Processing/Search_neurips_paper/head_scraped.png)

## **Sentence embeddings**

Using [SentenceTransformers](https://www.sbert.net/) we can get the embeddings of sentence, SentenceTransformers is a Python framework for getting text and image embeddings.

we have $$\textbf{title}$$ and $$\textbf{abstract}$$ as input, join and create a single input and get the embeddings using SentenceTransformers library, pretrained weights of Sci-BERT(**_pritamdeka/S-Scibert-snli-multinli-stsb_**) is used to get better representation of scientific papers.

## **Annoy**

Using embeddings, Annoy indexes papers based on the trees.

It will inspect up to search $$k$$ nodes during the query and gives the topk indexes.

Since we have embeddings, instead of word index, Annoy has method to search based on vector ie, _getnnsbyvector_ and returns the closest k items.

## **Web app**

I have created [web app](https://ravikumarmn-similar-research-papers-predictor-app-8zwxcb.streamlit.app/) using streamlit. It gives topk nearest neighbors of research papers.

![web_app_neurips.png](images/Machine Learning/Natural_Language_Processing/Search_neurips_paper/web_app_neurips.png)
