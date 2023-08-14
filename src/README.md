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

