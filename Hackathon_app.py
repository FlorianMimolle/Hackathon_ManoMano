import streamlit as st
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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
df_August = pd.read_csv(link1, low_memory=False)
df_September = pd.read_csv(link2, low_memory=False)
df_October = pd.read_csv(link3, low_memory=False)
df_November = pd.read_csv(link4, low_memory=False)
df = pd.concat([df_August,df_September,df_October,df_November])

table = st.sidebar.checkbox("Show DataFrame")
page = st.sidebar.radio("Page",("DataFrame analysis","Shipping Fees","Payment"))
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
        st.title("BAD COMMENTS\n(score ≤ 6)")
    with col2:
        st.title("GOOD COMMENTS\n(score ≥ 7)")
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
    st.title("ANALYSIS OF THE SHIPPING FEE IMPACT")

    col1,col2,col3 = st.columns([1,20,1])
    with col2:
        # Creation of dataframes from df_work to use as data to plot

        # Shipping_fees_df creation
        shipping_fees_df = df_work['shipping_fees_bucket'].value_counts().rename_axis('shipping_fees_bucket').to_frame('counts').reset_index()

        # Shipping_fees_bucket_df creation
        shipping_fees_bucket_df = df_work.groupby(["shipping_fees_bucket"]).agg({'score':'mean', 'shipping_fees_bucket':'count'})
        shipping_fees_bucket_df.rename(columns={shipping_fees_bucket_df.columns[1]: "count" }, inplace = True)
        shipping_fees_bucket_df.sort_values(by='count', ascending=False, inplace=True)
        shipping_fees_bucket_df.reset_index(inplace=True)

        # Definition of the figure and subplots
        sns.set(rc={'axes.facecolor':'silver', 'figure.facecolor':'silver'})
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        fig.suptitle('Correlation between shipping fees and score \n From August to November 2021')
        plt.subplots_adjust(wspace = 0.8, top=0.8)

        # PIE CHART SHIPPING FEES BUCKET
        sns.set_style("dark")
        sns.despine()
        ax[0].pie(shipping_fees_df['counts'], autopct='%1.1f%%', pctdistance = 1.1, textprops={'fontsize': 14},
            colors=['lime','chartreuse','aquamarine','darkturquoise','darkcyan','darkslateblue','mintcream'])
        ax[0].legend(labels=shipping_fees_df['shipping_fees_bucket'], title = "Shipping fees bucket", loc = 'center right',
            bbox_to_anchor=(1.6, 0.5))
        ax[0].set(title='Weight of shipping fees buckets')

        # BART AND LINE PLOT SCORE PER SHIPPING FEES BUCKET
        sns.set_style("dark")
        sns.despine()
        sns.set_style('ticks')
        ax2 = sns.barplot(ax=ax[1], data=shipping_fees_bucket_df, x='shipping_fees_bucket', y='count', color = 'mediumspringgreen')
        ax2.set(xlabel = "Shipping fees bucket", ylabel = "Number of transactions", title="Number of transactions per shipping fees bucket")
        ax2.tick_params('x', labelrotation=45)
        for bar in ax2.patches:
            ax2.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=9, xytext=(0, 8),
                   textcoords='offset points')
        ax3 = ax2.twinx()
        sns.lineplot(ax=ax3, data=shipping_fees_bucket_df, x='shipping_fees_bucket', y='score', color="blue", sort=False)
        ax3.set(ylabel = "Average score")
        ax3.set_ylim([7, 10])

        # Figure display
        st.pyplot(fig.figure)
    
if page == "Payment":
    st.title("ANALYSIS OF THE PAYMENT METHOD IMPACT")
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
    dfu.tags=dfu.tags.sort_values()
    dfupay = dfu[dfu["tags"]=="Difficulty Paying"]
    dfupaycount = dfupay.groupby(by="payment_method").count()[["id"]]
    dfupaycount.rename(columns={"id":"#difficulty Paying"}, inplace=True)
    dfupaycounttot = df.groupby(by="payment_method").count()[["id"]]
    dfupaycounttot.rename(columns={"id":"#total_transactions"}, inplace=True)
    dfupaycount = dfupaycount.join(dfupaycounttot, how="inner")
    for i in dfupaycount["#difficulty Paying"] :
        dfupaycount["% payment issues"] = (dfupaycount["#difficulty Paying"] / dfupaycount["#total_transactions"])*100
    dfupaycount["% payment issues"] = round(dfupaycount["% payment issues"],2)
    dfu_sort = dfupaycount[["% payment issues","#total_transactions"]].sort_values(by = "% payment issues", ascending = False).reset_index()

    fig = plt.figure(figsize=(10, 10))
    plt.bar(dfu_sort["payment_method"],dfu_sort["% payment issues"])
    plt.ylabel("Payment issues %")
    plt.title('Payment issues by payment method')
    
    col1, col2,col3 = st.columns([1,3,2])
    with col2:
        st.pyplot(fig.figure)
    with col3:
        a = dfu_sort[["payment_method","#total_transactions"]].set_index("payment_method").rename(columns = ["Transaction"])
        a
