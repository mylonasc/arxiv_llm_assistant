# ArxivHero

This is a hyper-specialized arxiv summarizer. I the source of inspiration was [this video](https://youtu.be/u4CRHtjyHTI?t=3303) of S. Wolfram speaking about LLM applications,
where he touched upon a common problem with the LLM research field: namely, there are too many papers being published every day and a customized way to summarize and read them is necessary. 

The engine can also take into account the "intent" of the user, not only "keywords". See for instance the prompts used in the doc generation engine:

```python

ENTERPRISE_USE_CASES_AND_INNOVATION = '''
            potential immediate enterprise use cases, as well as the claimed benefits of the original innovations
            '''

class DocGenerationEngine:
    def __init__(self, search_results, summary_focus = ENTERPRISE_USE_CASES_AND_INNOVATION):
        # def llm_summarizer(paper_text):

        self.search_results = search_results
        
        self.summary_focus_summ = ENTERPRISE_USE_CASES_AND_INNOVATION
        prompt_template_summ = """
            Write a very short (around 50 words) summary
            of the following paper title and abstract. Focus on {summary_focus} of the described techniques.:\n {paper_text}
            """
        
        prompt_template_intro = '''
            Write a short and engaging title (around 30 words) and introduction 
            (around 200 words) for a document that presents recent machine learning research news.
            Focus the intro on potential breakthroughs from the presented research. 
            
            Here is the text to use as reference:
            {input_text}
        
            wrap the title in an HTML <h1> tag and the intro in a <p> tag.
            Begin!
            '''

```

The system also includes a hand-engineered query + re-ranking engine, paired with a small ChatGPT-based summarizer + intro maker engine, that I created 
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
This is at a work-in-progress stage at the moment. Plan is to operationalize this somehow. 


