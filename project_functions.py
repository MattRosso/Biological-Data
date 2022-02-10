#!/usr/bin/env python
# coding: utf-8

# # Functions used in the project

# In[ ]:


from Bio import SearchIO
from Bio import SeqIO
from Bio.Blast import NCBIXML
import pandas as pd

import numpy as np
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
import copy

from collections import Counter

from chart_studio import plotly
import plotly.graph_objs as go

import igraph
from igraph import *

import pickle
import re
import string

from fisher import pvalue

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

from project_functions import *



# In[ ]:


def model_metrics(tn, tp, fp, fn):
    
    prec = 0
    rec = 0
    fscore = 0
    mcc = 0    
    
    prec = tp/(tp+fp)
    # print('Precision: ', prec)
    
    rec = tp/(tp+fn)
    # print('Recall: ', rec)
    
    if prec != 0 and rec != 0:
        fscore = (2*prec*rec)/(prec+rec)
        # print('F-Score: ', fscore)
    else:
        print('Precision and recall are both zero!')
    
    mcc = ((tp*tn)-(fp*fn))/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    # print('MCC: ', mcc)
    
    return(prec, rec, fscore, mcc)


# In[ ]:


def global_level(hmmsearch, psiblast, ground_truth_f, treshold_hmm = 0, treshold_pssm = 0):
    gt = set(ground_truth_f['Entry'])

    hmm = set(hmmsearch[hmmsearch['score'] > treshold_hmm]['acc2'])
    pssm = set(psiblast[psiblast['bit_score'] > treshold_pssm]['accession_id'])

    dim_swissprot = 565928
    metrics = []

    for name, data in zip(['hmm', 'pssm'], [hmm, pssm]):
        fn = len(gt - data)
        tp = len(gt.intersection(data))
        fp = len(data - gt)
        tn = dim_swissprot - tp
        # print(name, 'FN', fn)
        # print(name, 'TP', tp)
        # print(name, 'FP', fp)
        # print(name, 'TN', tn)
        
        
        metrics.append(model_metrics(tn, tp, fp, fn))
        
    return(metrics)


# In[ ]:


def get_ith_value(a_list, the_index):
  if len(a_list)<=the_index:
    return(-1)
  else:
    return(a_list[the_index])


# In[ ]:


def parsing_API_json(list_of_results_func):
  data = []
  for page in list_of_results_func:
    for protein in page['results']:
      name = protein['metadata']['accession']
      length = protein['metadata']['length']
      start = protein['entries'][0]['entry_protein_locations'][0]['fragments'][0]['start']
      end = protein['entries'][0]['entry_protein_locations'][0]['fragments'][0]['end']
      a = b = -1
      if len(protein['entries'][0]['entry_protein_locations']) > 1: #only one edge case
        a = protein['entries'][0]['entry_protein_locations'][1]['fragments'][0]['start']
        b = protein['entries'][0]['entry_protein_locations'][1]['fragments'][0]['end']
      data.append([name, length, start, end, a, b ])
  df = pd.DataFrame(data, columns=["accession", "length","start_gt","end_gt", "start_gt_2", "end_gt_2"])
  return(df)


# In[ ]:


def create_list(lenseq, p_i, p_f, p_i2, p_f2, m_i, m_f, m_i2 = -1, m_f2 = -1):
    
    valori_gt = []
    valori_m = []
    
    for i in range(lenseq):
        if (i >= p_i and i <= p_f) or (i >= p_i2 and i <= p_f2):
            valori_gt.append(1)
        else:
            valori_gt.append(0)
            
        if (i >= m_i and i <= m_f) or (i >= m_i2 and i <= m_f2):
            valori_m.append(2)
        else:
            valori_m.append(0)   
    
    return valori_gt, valori_m


# In[ ]:


def final_list_gt_hmm(row, gt_hmm):
    lenseq = int(gt_hmm['length'][row])
    
    gt_i = int(gt_hmm['start_gt'][row])
    gt_f = int(gt_hmm['end_gt'][row])
    gt_i2 = int(gt_hmm['start_gt_2'][row])
    gt_f2 = int(gt_hmm['end_gt_2'][row])
    
    m_i = int(gt_hmm['alisqfrom'][row])
    m_f = int(gt_hmm['alisqto'][row])
    m_i2 = int(gt_hmm['alisqfrom2'][row])
    m_f2 = int(gt_hmm['alisqto2'][row])
    
    gt, m = create_list(lenseq, gt_i, gt_f, gt_i2, gt_f2, m_i, m_f, m_i2, m_f2)
    fin = []
    for i in range(len(gt)):
        s = 0
        s = gt[i] + m[i]
        fin.append(s)
    
    return fin.count(3), fin.count(0), fin.count(2), fin.count(1)


# In[ ]:


def final_list_gt_pssm(row, gt_pssm):
    lenseq = int(gt_pssm['length'][row])
    
    gt_i = int(gt_pssm['start_gt'][row])
    gt_f = int(gt_pssm['end_gt'][row])
    gt_i2 = int(gt_pssm['start_gt_2'][row])
    gt_f2 = int(gt_pssm['end_gt_2'][row])
    
    m_i = int(gt_pssm['subject_start'][row])
    m_f = int(gt_pssm['subject_end'][row])
    
    gt, m = create_list(lenseq, gt_i, gt_f, gt_i2, gt_f2, m_i, m_f)
    fin = []
    for i in range(len(gt)):
        s = 0
        s = gt[i] + m[i]
        fin.append(s)
    
    return fin.count(3), fin.count(0), fin.count(2), fin.count(1)


# In[ ]:


def parsing_hmmer(hmm):
    path = 'data/predictions/{}.xml'.format(hmm)
    tree = ET.parse(path)
    root = tree.getroot()
    
    i = 0
    for element in root[0]:
      i = i+1
    # print("number of total protein is:", i)
    
    res = pd.DataFrame(columns =['name', 'acc2', 'score', 'evalue', 'alisqfrom_list', 'alisqto_list'])
    for j in range(i-1):
      try:
        name = root[0][j].attrib['name']
        acc2 = root[0][j].attrib['acc2']
        score = root[0][j].attrib['score']
        evalue = root[0][j].attrib['evalue']
        if True: #len(root[0][j])>1:
          alisqfrom_list = []
          alisqto_list = []
          for element in root[0][j]:     
            if element.attrib and element.tag == 'domains':
              alisqfrom_list.append(element.attrib['alisqfrom'])
              alisqto_list.append(element.attrib['alisqto'])
        res.loc[j] = {'name': name, 'acc2': acc2, 'score': score, 'evalue': evalue, 'alisqfrom_list': alisqfrom_list, 'alisqto_list': alisqto_list}
      except Exception as e:
        print("problematic entry at:", j)
        print(e)
        
    res['alisqfrom'] = res['alisqfrom_list'].apply(lambda x: get_ith_value(x, 0))
    res['alisqto'] = res['alisqto_list'].apply(lambda x: get_ith_value(x, 0))
    res['alisqfrom2'] = res['alisqfrom_list'].apply(lambda x: get_ith_value(x, 1))
    res['alisqto2'] = res['alisqto_list'].apply(lambda x: get_ith_value(x, 1))
    res['score'] = pd.to_numeric(res['score'])
    res['evalue'] = pd.to_numeric(res['evalue'])
    hmm_model = res[['name', 'acc2', 'score', 'evalue', 'alisqfrom', 'alisqto', 'alisqfrom2', 'alisqto2']]
    
    return(hmm_model)


# In[ ]:


def parsing_psiblast(pssm):
    
    blast_input = 'data/predictions/{}.xml'.format(pssm)
    data = []
    with open(blast_input) as f:
        blast_records = NCBIXML.parse(f)

        # Iterate Psiblast rounds
        for blast_record in blast_records:

            # Iterate query alignments
            query_id = blast_record.query
            for i, alignment in enumerate(blast_record.alignments):
                subject_id = alignment.title
                accession_id = alignment.accession

                for hsp in alignment.hsps:
                    data.append((query_id,
                                    subject_id,
                                    accession_id,
                                    blast_record.query_length,
                                    hsp.query,
                                    hsp.match,
                                    hsp.sbjct,
                                    hsp.query_start,
                                    hsp.query_end,
                                    hsp.sbjct_start,
                                    hsp.sbjct_end,
                                    hsp.identities,
                                    hsp.positives,
                                    hsp.gaps,
                                    hsp.expect,
                                    hsp.score))

    df = pd.DataFrame(data, columns=["query_id", "subject_id", "accession_id", "query_len",
                                  "query_seq", "match_seq", "subject_seq",
                                  "query_start", "query_end", 
                                  "subject_start", "subject_end", "identity", "positive", "gaps", "eval", "bit_score"])

    pssm_df = df[['accession_id', 'bit_score', 'eval', 'subject_start', 'subject_end']]
    
    return(pssm_df)


# In[ ]:


def residue_level_hmm(hmm_model, gt_df):

    final_gt_df = gt_df.loc[gt_df['DoesItContainIt'] == True].reset_index()[['accession', 'length', 'start_gt', 'end_gt', "start_gt_2", "end_gt_2"]]
    
    
    gt_hmm = pd.merge(left=final_gt_df, right=hmm_model, left_on='accession', right_on='acc2')
        
    TP = []
    TN = []
    FP = []
    FN = []

    for row in range(len(gt_hmm['accession'])):
        tp,tn,fp,fn = final_list_gt_hmm(row, gt_hmm)
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)

    return(model_metrics(np.sum(TN),np.sum(TP),np.sum(FP),np.sum(FN)))


# In[ ]:


def residue_level_pssm(pssm_df, gt_df):

    final_gt_df = gt_df.loc[gt_df['DoesItContainIt'] == True].reset_index()[['accession', 'length', 'start_gt', 'end_gt', "start_gt_2", "end_gt_2"]]
    
    
    gt_pssm = pd.merge(left=final_gt_df, right=pssm_df, left_on='accession', right_on='accession_id')
    
    TP = []
    TN = []
    FP = []
    FN = []

    for row in range(len(gt_pssm['accession'])):
        tp,tn,fp,fn = final_list_gt_pssm(row, gt_pssm)
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)

    return(model_metrics(np.sum(TN),np.sum(TP),np.sum(FP),np.sum(FN)))


# In[1]:


def make_annotations(pos, text, m, font_size=10, font_color='rgb(250,250,250)'):
    
    L=len(pos)
    
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=text[k], # or replace labels with a different list for the text within the circle
                x=pos[k][0], y=2*m-pos[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations


# In[ ]:


def get_go_from_swissprot(accession_func, dictionary_go_func):
    return(dictionary_go_func[accession_func][2])


# In[ ]:


def parse_obo(obo_file):
    """
    Parse the ontology (OBO format) and store into a
    dictionary, exclude obsolete terms
    
    graph = {
        <term_id>: {
            id: <term_id>, 
            name: <definition>, 
            is_a: [<parent_id>, ...] 
            is_obsolete: False, 
            namespace: <namespace>
        }
    }
    """
    graph = {}  # { term_id : term_object }
    obj = {}  # { id: term_id, name: definition, is_a: list_of_parents, is_obsolete: True, namespace: namespace }
    with open(obo_file) as f:
        for line in f:
            line = line.strip().split(": ")
            if line and len(line) > 1:
                # print(line)
                k, v = line[:2]
                if k == "id" and v.startswith("GO:"):
                    obj["id"] = v
                elif k == "name":
                    obj["def"] = v
                elif k == "is_a" and v.startswith("GO:"):
                    obj.setdefault("is_a", []).append(v.split()[0])
                elif k == "is_obsolete":
                    obj["is_obsolete"] = True
                elif k == "namespace":
                    obj["namespace"] = v
            else:
                if obj.get("id") and not obj.get("is_obsolete"):
                    if "is_a" not in obj:
                        obj["is_root"] = True
                    graph[obj["id"]] = obj
                    # print(obj)
                obj = {}
    return graph


# In[ ]:


def get_ancestors(graph, roots):
    """
    Build a dictionary of ancestors
    and calculate terms depth (shortest path)
    """
    depth = {}  # { term : depth }
    ancestors = {}  # { term : list_of_ancestor_terms }
    for node in graph:
        c = 0
        node_ancestors = []
        node_parents = graph[node].get("is_a")

        # Loop parents levels (parents of parents) until no more parents
        while node_parents:
            c += 1

            # Set root
            if node not in depth and roots.intersection(set(node_parents)):
                depth[node] = c

            # Add ancestors
            node_ancestors.extend(node_parents)

            # Update the list of parents (1 level up)
            node_parents = [term for parent in node_parents for term in graph[parent].get("is_a", [])]

        ancestors[node] = set(node_ancestors)
    return ancestors, depth


# In[ ]:


def get_children(ancestors):
    children = {}  # { node : list_of_children }, leaf terms are not keys
    for node in ancestors:
        for ancestor in ancestors[node]:
            children.setdefault(ancestor, set()).add(node)
    return children


# In[ ]:

def prosite_to_re(pattern):
    """convert a valid Prosite pattern into an re string"""
    _prosite_trans = str.maketrans("abcdefghijklmnopqrstuvwxyzX}()<>",
                                  "ABCDEFGHIJKLMNOPQRSTUVW.YZ.]{}^$")
    flg = (pattern[:2] == "[<")
    s = pattern.replace("{", "[^")
    s = s.replace(".", "")
    s = s.replace("-", "")
    s = s.translate(_prosite_trans)
    # special case "[<" and ">]", if they exist
    if flg:
        i = s.index("]")
        s = "(?:^|[" + s[2:i] + "])" + s[i+1:]
    if s[-2:] == "$]":
        i = s.rindex("[")
        s = s[:i] + "(?:" + s[i:-2] + "]|$)"
    elif s[-3:] == "$]$":
        i = s.rindex("[")
        s = s[:i] + "(?:" + s[i:-3] + "]|$)$"
    return s



# In[ ]:

def find_all_motif(entire_sequence, database, position_start_end):
  list_regex_position = {}
  for index, row in database.iterrows():
    regex_element = row['Regex']
    list_tuple_start_end = []
    for sequence_slice in position_start_end:
      sequence = entire_sequence[sequence_slice[0]:sequence_slice[1]]    
      regex_results_iterator = re.finditer(regex_element, sequence)
      for regex_results in regex_results_iterator:
        tuple_start_end = (regex_results.start()+1+sequence_slice[0],regex_results.end()+sequence_slice[0])
        list_tuple_start_end.append(tuple_start_end)
    if len(list_tuple_start_end) != 0: #if not list_tuple_start_end:
      list_regex_position[row['Accession']] = list_tuple_start_end
  return(list_regex_position)



# In[ ]:




