import arxiv 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from scipy.sparse._csr import csr_matrix
from typing import List, Dict, Tuple
from abc import ABC
from langchain import PromptTemplate, OpenAI, LLMChain
import enum 

import sys
sys.path.append('..')

## Special for this notebook - depends on relative paths:
def _get_helper_css_js():
    css = open('src/assets/style.css','r').read()
    js = open('src/assets/script.js','r').read()
    return css, js

def _get_openai_api_key():
    if 'OPENAI_API_KEY' in os.environ:
        openai_api_key = os.environ['OPENAI_API_KEY']
        return openai_api_key
    openai_api_key = open('../secret_openai_api_key.txt','r').read()[:-1]
    return openai_api_key

openai_api_key = _get_openai_api_key()


css, js = _get_helper_css_js()

## Some code to make pretty-printig HTML better:
s_pre =  '<html>'
s_pre += ' <head>'
s_pre += '  <style>' + css + '</style>'
s_pre += ' </head>'
s_pre += ' <body>'
## post:
s_post = '</body>'
s_post += '</html>'

def collapsible_button_html(title, content, score):
    if score is not None:
        s  = '   <button class="collapsible">(%2.3f) %s </button> '%(score, title)
    else:
        s  = '   <button class="collapsible">%s </button> '%( title)
    s += '   <div class="content">'
    s += '   <p>' + content + '</p>'
    s += '   </div>'
    return s

def _make_collapsible(title, content, score = None):
    s += s_pre
    s += collapsible_button_html(title, content, score)
    s += '<script>' + js + '</script>'
    s += s_post
    return s

def _make_collapsibles(titles, content_list, scores_list):
    s = s_pre
    for k,(t, c) in enumerate(zip(titles, content_list)):
        if scores_list is not None:
            s += collapsible_button_html(t,c,scores_list[k])
        else:
            s += collapsible_button_html(t,c)
    s += '<script>' + js +'</script>'
    s += s_post
    return s

def _get_arxiv_papers_for_query(query, num_papers = 200):
    """
    gets a list of arxiv papers according to some query
    """
    
    res = arxiv.Search(
      query,
      id_list=[],
      max_results = num_papers,
      sort_by = arxiv.SortCriterion.SubmittedDate,
      sort_order = arxiv.SortOrder.Descending
    )
    return res

def _get_arxiv_paper_list_text_data(arxiv_res):
    """
    Returns a list that contains the texts, given a list of arxiv results.
    """
    res_text_dat = [];
    query_results = []
    for r in arxiv_res.results():
        text_dat =  r.title +':\n\n' + r.summary
        res_text_dat.append(text_dat)
        query_results.append(r)
    return res_text_dat, query_results


def _topic_indices_from_topic_matrix(topic_matrix):
    inds = []
    for row in topic_matrix.T:
        inds.append(np.where(row)[0])
    return inds

def _get_topics_nmf_tfidf(
        texts,
        ntopics = 15,
        topic_accept_nmf_thresh = 0.1,
        topic_rel_q_thresh = 0.75
    ):
    """
    Gets a set of keywords using tf-idf, and simply non-negative matrix decomposition

    Args:
      texts  : a list of texts
      ntopics : number of topics
      topic_accept_nmf_thresh : the threshold above which to accept NMF components 
                      (the matrix is already usually sparse, but this helps making 
                      more intuitive sets of keywords for the papers)\
      
    """
    tfidf = TfidfVectorizer(stop_words='english')
    t = tfidf.fit_transform(texts)
    rr = NMF(n_components=ntopics).fit_transform(t.T)
    fnames = tfidf.get_feature_names_out()

    # get words for topics:
    topic_words = []
    for r in rr.T:
        topic_words.append(fnames[r>topic_accept_nmf_thresh])

    topic_rel = t @ rr
    q_v = np.quantile(topic_rel,topic_rel_q_thresh,1)
    # pplot.pcolor(topic_rel.T > q_v)
    topic_matrix = topic_rel.T>q_v
    topic_indices = _topic_indices_from_topic_matrix(topic_matrix)
    return  t, {'topic_indices' : topic_indices, 'topic_words' : topic_words ,'thresholded_topic_matrix' : topic_matrix}


def _second_level_topic_selection(paper_query_rel_scores, topic_data, q_thresh_val = 0.80):
    """Further filtering of topics based on paper-query relevance

    This creates a dictionary of topics that are relevant to the papers in the list.

    The paper_query_rel_scores can be computed (for instance) as follows: 
    
      `paper_query_rel_scores = emb_res @ enc_quer`

    """
    
    all_topic_inds = np.unique(np.stack(topic_data['topic_indices']))
    top_rel_scores = {}
    for topic_list, paper_score in zip(topic_data['topic_indices'], paper_query_rel_scores):
        for t in topic_list:
            if t not in top_rel_scores:
                top_rel_scores[t] = [paper_score, 1]
            else:
                top_rel_scores[t][0] += paper_score
                top_rel_scores[t][1] += 1
                
    for k in top_rel_scores.keys():
        avg = top_rel_scores[k][0] / top_rel_scores[k][1]
        top_rel_scores[k].append(avg)
    
    q_thresh = np.quantile([v[2] for k, v in top_rel_scores.items()], q_thresh_val)
    topic_filter = {k : v[2] >= q_thresh for k, v in top_rel_scores.items()}
    
    #contains "true" when this is a topic to be kept.
    kept_topics = {'indices' : [] , 'keywords' : []}
    discarded_topics = {'indices' : [] , 'keywords' : []}
    
    for t, b in topic_filter.items():
        if b:
            kept_topics['indices'].append(t)
            kept_topics['keywords'].append(topic_data['topic_words'][t])
        else:
            discarded_topics['indices'].append(t)
            discarded_topics['keywords'].append(topic_data['topic_words'][t])

    discarded_paper_inds, kept_paper_inds = [], []
    for k,p in enumerate(topic_data['topic_indices']):
        is_in_kept = False
        for i in p:
            if i in kept_topics['indices']:
                kept_paper_inds.append(k)
                is_in_kept = True
                break
                
        if not is_in_kept:
            discarded_paper_inds.append(k)
        
    return (kept_paper_inds, discarded_paper_inds), (kept_topics, discarded_topics)


class TopicModeler(ABC):
    def __init__(self):
        pass
    def __call__(self, v : List[str]):
        pass
        
class TFIDFNMFTopicModeler(TopicModeler):
    def __init__(
            self,
            ntopics = 5,
            topic_accept_nmf_thresh = 0.1,
            topic_rel_q_thresh = 0.75
        ):
        self.ntopics = ntopics
        self.topic_accept_nmf_thresh = topic_accept_nmf_thresh
        self.topic_rel_q_thresh = topic_rel_q_thresh
    
    def __call__(self, vals : List[str]) -> [csr_matrix, Dict] :
        return _get_topics_nmf_tfidf(
            vals,
            ntopics=self.ntopics,
            topic_accept_nmf_thresh=self.topic_accept_nmf_thresh,
            topic_rel_q_thresh=self.topic_rel_q_thresh
        )
        
class ArxivCustomRetrieval:
    """
    A hand-engineered retrieval engine, that 
    performs simple topic modeling and simple inner-product embedings-based
    topic and paper relevance determination (for filtering the most relevant papers)
    """
    def __init__(
            self, 
            topic_modeler : TopicModeler, 
            q_topic_thresh_val = 0.8,
            num_papers_query = 100,
            top_n_relevant = 10,
            embeding_model_str = 'thenlper/gte-base', **kwargs
        ):
        """
        A simple stateful wrapper to all the utility functions.

        Args:

            topic_modeler: an object that can return topics from text (in a speciffic format)
            topic_2nd_lv_quantile_thresh : after preliminary ID of the topics, using the embeddings of the retrieved texts, 
                           the IDed topics are re-evaluated for matching the initial query. This value is used to compute 
                           quantiles of topic-matching values (computed as the average of relevance score of topic-related
                           papers. 
            
        """
        
        self.topic_modeler = topic_modeler
        self.num_papers_query = num_papers_query
        model = SentenceTransformer(embeding_model_str)
        self.embedding_model = model
        self.q_topic_thresh_val = q_topic_thresh_val
        self.docs_embedded = False
        self.arxiv_papers_retrieved = False
        self._top_n_relevant = top_n_relevant # controls how many of the papers are actually printed/used

    def get_full_state(self):
        query_state = (self.text_res, self.query_res)
        embedding_state = (self._enc_quer, self._emb_res ,self._paper_query_rel_scores, self.docs_embedded)
        topic_state = (self.topic_matrix, self.topic_data)
        return {'topic_state' : topic_state, 'embedding_state' : embedding_state, 'query_state' : query_state} 
        
    def set_full_state(self, state):
        self.topic_matrix, self.topic_data = state['topic_state']
        self._enc_quer, self._emb_res, self._paper_query_rel_scores, self.docs_embedded  = state['embedding_state']
        self.text_res, self.query_res = state['query_state']
        
        
    def run(self, query):
        """
        Full run:
         1. running the retrieval from arxiv
         2. performing topic modeling with TFIDF and NMF 
         3. reducing the topics to the most discriminative (through thresholding)
         4. embeding the retrieved documents (using a transformer model)
         5. computing the relevance of the retrieved documents with the provided query
         6. finding the mean topic relevance (by summing per-
           topic instance how relevant the documents that were assigned that topic are)
         7. setting (through thresholding) the most relevant topics
         8. discarding/keeping according to topic relevance the corresponding papers
        """
        # 1. 
        self.execute_query(query)
        # 2. + 3.
        self.get_topic_data(self.text_res)
        # 4. + 5. 
        self.embed_docs(self.text_res, query)
        # paper_query_rel_scores = self._emb_res @ self._enc_quer
        # 6. 7. 8. 
        (kept_paper_inds, discarded_paper_inds), (kept_topics, discarded_topics) = _second_level_topic_selection(
            self._paper_query_rel_scores, 
            self.topic_data,
            q_thresh_val = self.q_topic_thresh_val
        )
        
        self.kept_paper_inds, self.discarded_paper_inds = kept_paper_inds, discarded_paper_inds
        self.kept_topics, self. discarded_topics = kept_topics, discarded_topics        

    def get_kept_papers_text(self):
        return [self.text_res[k] for k in self.kept_paper_inds]

    def get_kept_papers_results(self):
        return [self.query_res[k] for k in self.kept_paper_inds]
        
    def get_kept_paper_arxiv_query_res(self):
        return [self.query_res[k] for k in self.kept_paper_inds]

    def embed_docs(self, text_res, interests_query):
        if not self.docs_embedded:
            emb_res = self.embedding_model.encode(text_res)
            enc_quer = self.embedding_model.encode(interests_query)
            self._enc_quer = enc_quer
            self._emb_res = emb_res
            self._paper_query_rel_scores = emb_res @ enc_quer
            self.docs_embedded = True

    def get_topic_data(self, text_list : List[str]):
        """
        Creates a set of topics (as a sparse matrix and a vector of topic indices)
        
        See also:
          `TFIDF_NMF_TopicModeler`
        """
        self.topic_matrix, self.topic_data = self.topic_modeler(text_list)
        
    def execute_query(self, query):
        """
        Executes the query and stores the results in the object.
        """
        res = _get_arxiv_papers_for_query(query, num_papers=self.num_papers_query)
        text_res, query_res = _get_arxiv_paper_list_text_data(res)
        self.text_res, self.query_res = text_res, query_res
        
    def get_most_relevant_inds(self, top_n = None):
        if top_n is None:
            top_n = self._top_n_relevant
        return np.argsort(-self._paper_query_rel_scores)[:top_n]
        
    def _repr_html_(self):
        """returns HTML ready to render or add as a component, that contains title and abstract from the arxiv papers.
        """
        argsort_scores = np.argsort(-self._paper_query_rel_scores)
        query_results = self.query_res
        titles, contents, scores = [], [], []
        for idx in argsort_scores[:self._top_n_relevant]:
            q = self.query_res[idx]
            score = self._paper_query_rel_scores[idx]
            titles.append(q.title)
            contents.append(q.summary)
            scores.append(score)
            
        s = _make_collapsibles(titles, contents, scores)
        return s


def _make_html_from_query_res(query_res, summary):
    entry_id = query_res.entry_id
    s = ''
    s += '<div class="content">\n<h4><a href=' + entry_id + '>' + query_res.title
    s += ' (' + entry_id.split('/')[-1] + ')'  + '</a> </h4>\n' 
    s += '<p>' + summary + '</p>\n'
    s += '</div>'
    return s

##### prompts ##########
class SummaryFocus(enum.Enum):
    """
    The summary focus also corresponds 
    to a part of the summarization 
    prompt.
    """
    ENTERPRISE_USE_CASES_AND_INNOVATION = '''
        potential immediate enterprise use cases, as well as the claimed benefits of the original innovations
        '''
    ACADEMIC_RESEARCH = '''
        the potential for the presented benefits to create a lasting impact in academic research
        '''

class FlatDomainOntology(enum.Enum):
    """
    The string values of this enum are 
    used in the prompt to specialize the intro.
    The point is for the intro of the article 
    to share the intent of the reader and be somehow 
    close to the reason for the creation of the report,
    so chose this wisely!
    """
    ML_RESEARCH_NEWS_DOC_ONTOLOGY = 'machine learning research'
    RESEARCH_ONTOLOGY = 'research'
    BUSINESS = 'enterprise business development'
    GEOPOLITICS_AND_MACROECONOMICS = 'outlook on financial research considering geopolitics and macroeconomics'


class DocGenerationEngine:
    def __init__(
                self,
                search_results,
                summary_focus = SummaryFocus.ENTERPRISE_USE_CASES_AND_INNOVATION,
                doc_ontology_text = FlatDomainOntology.RESEARCH_ONTOLOGY
            ):

        self.search_results = search_results
        self.doc_ontology_text = doc_ontology_text.value
        self.summary_focus_summ = summary_focus.value
        
        prompt_template_summ = """
            Write a very short (around 50 words) summary
            of the following paper title and abstract. Focus on {summary_focus} of the described techniques.:\n {paper_text}
            """
        
        prompt_template_intro = '''
            Write a short and engaging title (around 30 words) and introduction 
            (around 200 words) for a document that presents recent {doc_ontology}.
            Focus the intro on potential breakthroughs from the presented text. 
            
            Here is the section of text to use for the summary:
            {input_text}
        
            wrap the title in an HTML <h1> tag and the intro in a <p> tag.
            Begin!
            '''
        
        self.prompt_intro = PromptTemplate.from_template(prompt_template_intro)
        self.prompt_summarization = PromptTemplate.from_template(prompt_template_summ)
        
        self.llm = OpenAI(temperature=0.1,openai_api_key = openai_api_key)
        
        self.llm_chain_summarizer = LLMChain(
            llm=self.llm,
            prompt=self.prompt_summarization,
        )
        self.llm_chain_intro_maker = LLMChain(
            llm = self.llm, 
            prompt = self.prompt_intro
        )
        self.paper_summaries = None
        self.intro = None
    
    def _paper_abstract_summarizer(self,p):
        # def llm_summarizer(paper_text):
        return self.llm_chain_summarizer.run(summary_focus = str(self.summary_focus_summ), paper_text = p)
    
    def _document_intro_maker(self, paper_titles_concat):
        return self.llm_chain_intro_maker.run(input_text = paper_titles_concat, doc_ontology = str(self.doc_ontology_text))

    def make_document(self, add_body_and_html = True, add_css = True):
        from tqdm import tqdm
        ## Create very brief summaries for each paper:
        paper_summaries = []
        
        sr = self.search_results
        most_rel_query_res = [sr.query_res[k] for k in sr.get_most_relevant_inds()]
        most_rel_text_res = [sr.text_res[k] for k in sr.get_most_relevant_inds()]
        if self.paper_summaries is None:
            for p in tqdm(most_rel_text_res):
                paper_summaries.append(self._paper_abstract_summarizer(p))
            self.paper_summaries = paper_summaries
            
        if self.intro is None:
            self.paper_summaries_for_render =  [_make_html_from_query_res(q, summ) for q, summ in zip(most_rel_query_res, self.paper_summaries)]
            self.intro = '<div class="intro content">' + self._document_intro_maker(''.join(paper_summaries)) +' </div>'


        s = ''
        if add_body_and_html:
            s += '<html>'
            if add_css:
                s += '<head>'
                s +='<style>'
                s += css
                s +='</style>'
                s += '</head>'
            s += '<body>'

        s += self.intro + '\n'.join(self.paper_summaries_for_render)

        if add_body_and_html:
            s += '</body>'
            s += '</html>'
        return  s
            

