#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: saj, manuel
"""

#Morphological processing for boosting the performance of German neural named entity recognition

import spacy
import timeit
from tqdm import tqdm
from multiprocessing import cpu_count

# In[1]: Conversion of raw text corpus
def process_corpus(fileIn="Leipzig40MT2010", threads=cpu_count()):
    """
    Reads the given raw text corpus and converts it to their
    lemma, lemmapos versions
    """
    print ('Number of CPUs/threads:'+ str(threads))
    fIn = open('data/'+fileIn,'r+')
    fOut_lemma = open('out/'+fileIn+'.lemma', 'w')
    fOut_lemmapos = open('out/'+fileIn+'.lemmapos', 'w')
    big_text = fIn.readlines()
    fIn.close()
    
    nlp = spacy.load('de', disable=['parser','ner'])
    lines_lemma = []
    lines_lemmapos = []
    block_size = 100000
    for i, doc in enumerate(nlp.pipe(big_text, batch_size=100, n_threads=threads)):
        lines_lemma.append(doc[:].lemma_)
        lines_lemmapos.append(" ".join([(token.lemma_+'_'+token.tag_) for token in doc if not token.is_space]))
        if (i % block_size) == 0:
            print (str(i) + ' lines processed.')
    
    del big_text
    fOut_lemma.write("\n".join(lines_lemma)); fOut_lemma.write("\n")
    fOut_lemmapos.write("\n".join(lines_lemmapos)); fOut_lemmapos.write("\n")
    fOut_lemma.close(); fOut_lemmapos.close()

# In[2]: Conversion of train/test/dev datasets prior to any neural training process
def process_trainingData(fileIn, threads=cpu_count()):
    """
    Reads the given file in CoNLL 4-column format and converts it to their 
    lemma, lemmapos versions
    """
    fIn = open('data/' + fileIn, 'r+', encoding="utf8")
    inText = fIn.readlines()
    fIn.close()

    text = []
    lemmata = []
    pos = []
    entities = []
    for line in inText:
        split = line.split(" ")
        text.append(split[0])
        if len(split) == 4:
            lemmata.append(split[1])
            pos.append(split[2])
            entities.append(split[3].replace("\n", ""))
        else:
            lemmata.append("")
            pos.append("")
            entities.append("")

    del inText

    print('Tagging ' + fileIn + '..')
    nlp = spacy.load('de', disable=['parser', 'ner'])
    with open('out/' + fileIn + ".lemma", 'w', encoding="utf8") as fOut_lemma, open('out/' + fileIn + ".lemmapos", 'w', encoding="utf8") as fOut_lemmapos:
        for i, doc in tqdm(enumerate(nlp.pipe(text, batch_size=100, n_threads=threads)), total=len(text)):
            ent = entities[i]
            if doc.__len__() > 1:
                fOut_lemma.write(text[i] + ' ' + text[i] + ' ' + pos[i] + ' ' + ent + '\n')
                fOut_lemmapos.write(text[i] + '_' + pos[i] + ' ' + text[i] + ' _ ' + ent + '\n')
            else:
                for token in doc:
                    if not token.is_space:
                        tok = token.orth_
                        lemma = token.lemma_ if not token.lemma_ == "" else lemmata[i] if not lemmata[i] == "" else ""
                        if lemma.__contains__(" "):
                            lemma = token.orth_
                        tag = token.tag_ if not token.tag_ == "" else pos[i] if not pos[i] == "" else ""
                        fOut_lemma.write(lemma + ' ' + tok + ' ' + tag + ' ' + ent + '\n')
                        fOut_lemmapos.write(lemma + '_' + tag + ' ' + tok + ' _ ' + ent + '\n')
                    else:
                        fOut_lemma.write(token.orth_)
                        fOut_lemmapos.write(token.orth_)

    del text

def check_spaces(file, endings = [".lemma", ".lemmapos"]):
    """
    Checks the format of converted datasets
    """
    print("Checking " + file + " for additional spaces..")
    for appendix in endings:
        file_name = 'out/' + file + appendix
        file_in = open(file_name, 'r+', encoding="utf8")
        text = file_in.readlines()
        file_in.close()
        for i, line in tqdm(enumerate(text), total=len(text)):
            split = line.split(" ")
            if len(split) > 4:
                raise ValueError("Found more than three spaces in " + file_name + " at line " + i + ".")
                
# In[3]: Main  
if __name__ == "__main__":
    #1. text corpus
    start_time = timeit.default_timer()
    process_corpus()
    print ("Processing of text corpus completed! Overall time:"); print(start_time-timeit.default_timer())
    files = ["deu.train", "deu.dev", "deu.test"] # German datasets from CoNLL 2003 shared task
    for f in files:
        #2. training data
        process_trainingData(f)
        #3. check format
        check_spaces(f)