# updates the website with the latest news

# 1. run the document creation

import datetime
from bs4 import BeautifulSoup
today = datetime.datetime.today()

KEYWORDS='large language models,efficient transformers,graph neural networks,qlora'
TOP_N=10
Q_TOPICS=0.9
date_string_html = today.strftime('%d%m%y')
date_string = today.strftime('%a %d %b %y')
FNAME='outputs/daily/my_ml_papers_'+date_string_html+ '.html'

from arxivhero import _get_document_last_days_arxiv

if __name__=='__main__':

    # Make the document:
    doc = _get_document_last_days_arxiv(KEYWORDS.split(','), q_topic_thresh=Q_TOPICS, top_n_relevant=TOP_N)

    s = '<html>'
    s += '<head>'
    s += '</head>'
    s += doc
    s += '<body>'
    s += '</body>'
    s += '</html>'
    # save at correct place:
    with open(FNAME, 'w') as f:
        f.write(s)

    # add to index.html:
    soup = BeautifulSoup(open('index.html','r').read(),'html.parser')
    t = soup.new_tag('li')
    a = soup.new_tag('a')
    date_string_html = '%02i'%today.day +  '%02i'%today.month + ('%02i'%today.year)[-2:]

    a.attrs['href'] = 'outputs/daily/my_ml_papers_%s.html'%date_string_html
    a.string = 'My top ML papers - %s'%date_string
    t.append(a)
    soup.select('li')[-1].insert_after(t)

    open('index.html','w').write(str(soup))
    print('updated the files in output and the index.html')




