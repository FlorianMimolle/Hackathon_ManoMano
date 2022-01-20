import streamlit as st
import string 
import re
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

def textualisation(df):
    result = []
    for comment in df["clean"]:
        for mot in re.findall("\S+",str(comment)):
            result.append(mot)
    result = " ".join(result)
    return result

def couleur_bad(*args, **kwargs):
    import random
    return "rgb(255, 0, {})".format(random.randint(0, 150))

def couleur_good(*args, **kwargs):
    import random
    return "rgb({}, 255, 50)".format(random.randint(0, 150))

#Entête :
link1 = "https://raw.githubusercontent.com/FlorianMimolle/Hackathon_ManoMano/main/2022-01WildCodeSchoolxManoMano_CESDataSet-August2021.csv"
link2 = "https://raw.githubusercontent.com/FlorianMimolle/Hackathon_ManoMano/main/2022-01%20Wild%20Code%20School%20x%20ManoMano_%20CES%20Data%20Set%20-%20September%202021.csv"
link3 = "https://raw.githubusercontent.com/FlorianMimolle/Hackathon_ManoMano/main/2022-01%20Wild%20Code%20School%20x%20ManoMano_%20CES%20Data%20Set%20-%20October%202021.csv"
link4 = "https://raw.githubusercontent.com/FlorianMimolle/Hackathon_ManoMano/main/2022-01%20Wild%20Code%20School%20x%20ManoMano_%20CES%20Data%20Set%20-%20November%202021.csv"
df_August = pd.read_csv(link1)
df_September = pd.read_csv(link2)
df_October = pd.read_csv(link3)
df_November = pd.read_csv(link4)
df = pd.concat([df_August,df_September,df_October,df_November])

table = st.sidebar.checkbox("Show DataFrame")
page = st.sidebar.radio("Page",("DataFrame analysis","Shipping Fees","Connexion"))
pays = st.sidebar.radio("Filter Country",('all','GB', 'FR', 'IT', 'DE', 'ES'))
b2b = st.sidebar.radio("Type of customers",('all','b2b',"b2c"))

if pays == "all":
    df = df
if pays == "GB":
    df = df[df["platform"]=="GB"]
if pays == "FR":
    df = df[df["platform"]=="FR"]
if pays == "IT":
    df = df[df["platform"]=="IT"]
if pays == "DE":
    df = df[df["platform"]=="DE"]
if pays == "ES":
    df = df[df["platform"]=="ES"]   

if b2b == "all":
    df_work = df
if b2b == "b2b":
    df_work = df[df["is_b2b"]==True]
if b2b == "b2c":
    df_work = df[df["is_b2b"]==False]

if table:
    df_work
if page == "DataFrame analysis":
    st.title("INTRODUCTION")
    
    col1,col2,col3 = st.columns([1, 3,2])
    with col2:
        df_bad = df_work[df_work["score"]<7]
        a = round(len(df_bad)*100/len(df),2)
        b = 100-a
        fig = plt.figure()
        plt.pie([a,b],
				colors = ["red","green"],
				labels = [f"Bad score\n{a}%",f"Good score\n{b}%"])
        plt.title("Repartition of score")
        st.pyplot(fig.figure)
    
    col1,col2 = st.columns(2)
    with col1:
        st.title("BAD COMMENTS (score ≤ 6)")
    with col2:
        st.title("GOOD COMMENTS (score ≥ 7)")
    mot_bad = pd.read_csv("https://raw.githubusercontent.com/FlorianMimolle/Hackathon_ManoMano/main/df_NLP_bad%20(1)", sep = ",")
    mot_good = pd.read_csv("https://raw.githubusercontent.com/FlorianMimolle/Hackathon_ManoMano/main/df_NLP_good%20(1)", sep = ",")
   	
    if pays == "all":
        mot_bad = mot_bad
        mot_good = mot_good
    if pays == "GB":
        mot_bad = mot_bad[mot_bad["platform"]=="GB"]
        mot_good = mot_good[mot_good["platform"]=="GB"]
    if pays == "FR":
        mot_bad = mot_bad[mot_bad["platform"]=="FR"]
        mot_good = mot_good[mot_good["platform"]=="FR"]
    if pays == "IT":
        mot_bad = mot_bad[mot_bad["platform"]=="IT"]
        mot_good = mot_good[mot_good["platform"]=="IT"]
    if pays == "DE":
        mot_bad = mot_bad[mot_bad["platform"]=="DE"]
        mot_good = mot_good[mot_good["platform"]=="DE"]
    if pays == "ES":
        mot_bad = mot_bad[mot_bad["platform"]=="ES"]   
        mot_good = mot_good[mot_good["platform"]=="ES"] 
    
    if b2b == "all":
        df_work_bad = mot_bad
        df_work_good = mot_good
    if b2b == "b2b":
        df_work_bad = mot_bad[mot_bad["is_b2b"]==True]
        df_work_good = mot_good[mot_good["is_b2b"]==True]
    if b2b == "b2c":
        df_work_bad = mot_bad[mot_bad["is_b2b"]==False]
        df_work_good = mot_good[mot_good["is_b2b"]==False]
        
    texte_bad = textualisation(df_work_bad)
    texte_good = textualisation(df_work_good)
    
    wordcloud = WordCloud(width=480, 
						height=480, 
						random_state=1,
						max_font_size=200, 
						min_font_size=10,
						collocations=False,
						background_color = "black",
      					max_words=20)
    fig, ax = plt.subplots(figsize = (16,10), 
						nrows = 1,
						ncols = 2)
    fig.patch.set_facecolor('xkcd:black')
    wordcloud.generate(texte_bad)
    ax[0].imshow(wordcloud.recolor(color_func = couleur_bad))
    ax[0].set_title("commentaire mauvais")
    ax[0].axis("off")
    ax[0].margins(x=0, y=0)
    
    wordcloud.generate(texte_good)
    ax[1].imshow(wordcloud.recolor(color_func = couleur_good))
    ax[1].set_title("Bons commentaires")
    ax[1].axis("off")
    ax[1].margins(x=0, y=0)
    
    st.pyplot(fig.figure)
    
    col1,col2,col3 = st.columns([1, 3,2])
    with col2:
        dfu=df_work[~df_work.tags.isna()].reset_index()
        dfu=(dfu.set_index(['index', 'id', 'comment', 'original_comment', 'score', 'data_scale',
		'data_source', 'created_at', 'date', 'day', 'month', 'is_mf', 'device',
		'family', 'is_b2b', 'reason', 'browser', 'zipcode', 'category',
		'language', 'platform', 'provider', 'first_order', 'nb_articles',
		'csat_presales', 'shipping_fees', 'bv_transaction', 'payment_method',
		'operating_system', 'last_paid_channel', 'has_presales_contact',
		'shipping_fees_bucket', 'bv_transaction_bucket',
		'has_manodvisor_contact', 'themes']).apply(lambda x: x.str.split(';').explode()).reset_index())
        dfu.tags=dfu.tags.apply(lambda x: x.replace("Detractor '"," Detractor"))
        dfu.tags=dfu.tags.apply(lambda x: x.replace(" Detractor -"," Detractor-"))
        dfu.tags=dfu.tags.apply(lambda x: x.strip())
        dfu.tags=dfu.tags.apply(lambda x: x.replace("Detractor- ",""))
        dfu_sort = dfu["tags"].value_counts().reset_index()
        fig = plt.figure()
        plt.bar(dfu_sort["index"],dfu_sort["tags"])
        plt.xticks(rotation='vertical')
        plt.title("Number of tags")
        st.pyplot(fig.figure)
        
if page == "Shipping Fees":
    print("ok")
    
if page == "Connexion":
    print("ok")