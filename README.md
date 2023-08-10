# ArxivHero

This is a hyper-specialized arxiv summarizer. 

It includes a hand-engineered query + re-ranking engine, paired with a small ChatGPT-based summarizer + intro maker engine, that I created 
to be up to date with current research.

## Usage
Currently this contains just a notebook with everything needed (arxiv query manager, topic modeler, topic filtering, re-ranker, embedder, generative model language engine etc).

The lines that actually produce the output are the following (see "results" section in notebook)
```python
interests_query = "llm chatgpt efficient inference"
sr = ArxivCustomRetrieval(topic_modeler=TFIDFNMFTopicModeler(), q_topic_thresh_val=0.5, top_n_relevant=10)
sr.run(interests_query)
d = DocGenerationEngine(sr)
doc = d.make_document()
```

It will probably be extended to support more sources and do some more OSINT on the retrieved papers.

## The system performs the following
1. Retrieve recent arxiv papers according to query
    * using `arxiv` library
3.  find the topics of the retrieved papers
    * using e.g., NMF + tf-Idf
    * make intuitive subsets of the words describing the topics using further thresholding.
4.  compute the relevance of all the papers to the keywords (using embedings)
    * some heuristics with thresholding etc
5.  discard papers and corresponding to the topics with low mean relevance
6.  find the final relevant papers
7.  creates a summary of the abstracts of these papers
8.  Creates a title and introductory summary for the whole document
9.  Collates the text created to a final document

## Examples 
See the outputs folder.

## Limmitations
It may be hacky at points because I ...hacked it in an afternoon and a lunchbreak.


