from vicuna_setup import vicuna_inference
import pandas as pd

test_data = pd.read_csv("/home/medical-llama/Pubmedqa/pubmedqa/data/pubmedqa_test.csv")


def pubmed_template(Question, Context,id):
    if id==0:
        prompt_template=f"""

    Respond with ""yes , no , maybe "" using the context & the question provided.    

    ###
            
    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are dogs trainable?

    Answer: yes

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Can dogs be trained to detect and alert individuals to the presence of specific allergens?

    Answer: maybe

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are Dogs highly introverted & reserved ?

    Answer: no 

    ###

    Context:
    {Context}

    Question:
    {Question}

    Answer: 
    
    """
        return prompt_template
    
    if id==1:
        
        prompt_template=f"""

    Respond with ""yes , no , maybe "" using the context & the question provided.    

    ###
            
    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are dogs trainable?

    Answer: yes

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are Dogs highly introverted & reserved ?

    Answer: no     

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Can dogs be trained to detect and alert individuals to the presence of specific allergens?

    Answer: maybe

    ###

    Context:
    {Context}

    Question:
    {Question}

    Answer: 
    """
        return prompt_template
    
    if id==2:

        prompt_template=f"""

    Provide a single response of "yes," "no," or "maybe" based on the given context and question.
    
    ###
            
    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are dogs trainable?

    Answer: yes

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Can dogs be trained to detect and alert individuals to the presence of specific allergens?

    Answer: maybe

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are Dogs highly introverted & reserved ?

    Answer: no 

    ###

    Context:
    {Context}

    Question:
    {Question}

    Answer: 
    """
        return prompt_template
    
    if id==3:

        prompt_template=f"""

     The objective is to answer research questions with "yes," "no," or "maybe" based on the corresponding abstracts. The abstracts contain valuable information that can address specific research inquiries. For example:

    ###
            
    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are dogs trainable?

    Answer: yes

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Can dogs be trained to detect and alert individuals to the presence of specific allergens?

    Answer: maybe

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are Dogs highly introverted & reserved ?

    Answer: no 

    ###

    Context:
    {Context}

    Question:
    {Question}

    Answer: 
    """
        return prompt_template
    
    if id==4:
        prompt_template=f"""

    your task is to answering research questions relating to medical domain using yes/no/maybe responses. Answer as yes if the context supports the question , answer as no if the context does not support the question , answer as maybe if the context is not clear enough to answer the question.      

    ###
            
    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are dogs trainable?

    Answer: yes

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Can dogs be trained to detect and alert individuals to the presence of specific allergens?

    Answer: maybe

    ###

    Context: Research involving dogs has been instrumental in advancing our understanding of various scientific fields. Dogs are highly trainable and sociable animals.

    Question: Are Dogs highly introverted & reserved ?

    Answer: no 

    ###

    Context:
    {Context}

    Question:
    {Question}

    Answer: 
    """
        return prompt_template
    

def output_preprocessing(dict_ans):

    for i in dict_ans.keys():

        l_=dict_ans[i].split(" ")
        
        if l_[0]=="" and len(l_)>1:
            dict_ans[i]=l_[1].lower()    
            
        else:
            dict_ans[i]=l_[0].lower()    

        if "yes" in dict_ans[i]:
            dict_ans[i] = "yes"
        elif "no" in dict_ans[i]:
            dict_ans[i] = "no"
        elif "maybe" in dict_ans[i]:
            dict_ans[i] = "maybe"
        else:
            pass

    return dict_ans    
    

def model_pubmed_evaluator(model,tokenizer,config,id_,df=test_data):
    ans_dict={}
    for i in range(df.shape[0]):
        pmid=str(df['ID'][i])
        ans=vicuna_inference(pubmed_template(Question=df['Question'][i],Context=df['Context'][i],id=id_),model=model,tokenizer=tokenizer,config=config)
        ans_dict[pmid]=ans

        ans_dict=output_preprocessing(ans_dict)
    return ans_dict
