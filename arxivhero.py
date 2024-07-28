import os
from src.arxiv_hero import ArxivCustomRetrieval, TFIDFNMFTopicModeler, DocGenerationEngine
from src.arxiv_hero import FlatDomainOntology, SummaryFocus
from typing import List
import argparse
import logging


parser = argparse.ArgumentParser('Create a webpage containing a digest of recent arxiv papers according to some keywords.')
parser.add_argument('--keywords',type=str,nargs='+',
        help= 'keywords - include in quotes, separate with commas')
parser.add_argument('--top-n-relevant',default = 10, type=int,help = 'Number or relevant papers to retrieve.')
parser.add_argument('--topic-thresh',default=0.9, help='Quantile threshold for keeping/discarding topics. Lower values may allow irrelevant automatically identified topic keywords to be included. Make this low for potentially larger variety, keep this to high values for only using topics that seem relevant to the keywords provided.')

parser.add_argument('--out', type=str, help = 'output file (HTML doc)')

# interests_query = "llm chatgpt efficient inference"
def _get_document_last_days_arxiv(keywords : List[str], q_topic_thresh = 0.9 , top_n_relevant = 5):
    interests_query = ','.join(keywords) # 'finance china interest rates oil'
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s | %(levelname)s %(message)s" ,
        filename='arxiv_hero.log'
    )
    logging.info("- running querry with interests query: " + interests_query)
    ranked_search_results = ArxivCustomRetrieval(
            topic_modeler=TFIDFNMFTopicModeler(), 
            q_topic_thresh_val=q_topic_thresh,
            top_n_relevant=top_n_relevant
    )

    ranked_search_results.run(interests_query)

    logging.info("- running the LLM-based doc generation engine")
    try:
        d = DocGenerationEngine(
          ranked_search_results,
          doc_ontology_text=FlatDomainOntology.ML_RESEARCH_NEWS_DOC_ONTOLOGY,
          summary_focus=SummaryFocus.ACADEMIC_RESEARCH
        )
        logging.info("  doc generation ok")
    except:
        logging.error(" - doc generation failed!")
    return d.make_document()


if __name__ == '__main__':
    args = parser.parse_args()
    kwords = args.keywords[0].split(',')
    fname = args.out

    doc = _get_document_last_days_arxiv(kwords, q_topic_thresh = args.topic_thresh, top_n_relevant=args.top_n_relevant)
    s = '<html>'
    s += '<head>'
    s += '</head>'
    s += doc
    s += '<body>'
    s += '</body>'
    s += '</html>'
    with open(fname, 'w') as f:
        f.write(s)

