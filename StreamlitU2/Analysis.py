import networkx as nx
import streamlit as st
import pandas as pd
import seaborn as sns
import streamlit_pandas as sp
from adjustText import adjust_text

import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv(file, header=None, names=["Player_ID", "Game_title", "Behavior", "Hours", "Ignore"], nrows=2500,dtype={'Player_ID': str})
    return df

file = "./steam-200k.csv"
df = load_data()
df.drop("Ignore", inplace=True, axis=1)

# Ira analisar o df e ira agrupar por Player_ID e Game_title, e ira pegar o valor de "play" se existir, se nao ira pegar o valor de "purchase". basicamente resumindo o df
df_filtred = df.groupby(["Player_ID", "Game_title"]).apply(lambda x: x[x["Behavior"] == "play"] if "play" in x["Behavior"].values else x[x["Behavior"] == "purchase"]).reset_index(drop=True)

B = nx.Graph()

players = df_filtred['Player_ID'].unique()
games = df_filtred['Game_title'].unique()

B.add_nodes_from(players, bipartite=0)
B.add_nodes_from(games, bipartite=1)

for index, row in df_filtred.iterrows():
    B.add_edge(row['Player_ID'], row['Game_title'], weight=row['Hours'])

st.title('Steam Game/Players Analysis')
st.write('This is a simple analysis of the Steam dataset for network analysis exercice')
st.write('Choose a option on the sidebar to appear below')

if st.sidebar.button('Show unfiltred dataset'):
    st.write(df_filtred)

if st.sidebar.button('Information about the columns'):
    p, g, b , h= st.columns(4)
    p.write(df_filtred['Player_ID'].describe())
    g.write(df_filtred['Game_title'].describe())
    b.write(df_filtred['Behavior'].describe())
    h.write(df_filtred['Hours'].describe())


if st.sidebar.button('Show a sample of the complete graph'):
    steam_sorted = df.sort_values(by=['Behavior'], ascending=False)

    filtered_steam = steam_sorted.drop_duplicates(subset=['Player_ID', 'Game_title'])

    filtered_steam_sample = filtered_steam.sample(n=200)

    B = nx.Graph()

    players = filtered_steam_sample['Player_ID'].unique()
    games = filtered_steam_sample['Game_title'].unique()

    B.add_nodes_from(players, bipartite=0)
    B.add_nodes_from(games, bipartite=1)

    for index, row in filtered_steam_sample.iterrows():
        B.add_edge(row['Player_ID'], row['Game_title'])

    plt.figure(figsize=(20, 15))

    pos = nx.spring_layout(B, k=0.3)

    nx.draw_networkx_nodes(B, pos, nodelist=players, node_color='lightblue', node_size=50, label='Players')
    nx.draw_networkx_nodes(B, pos, nodelist=games, node_color='lightgreen', node_size=50, label='Games')

    nx.draw_networkx_edges(B, pos, alpha=0.5)

    #nome azul para os steamIds (jogadores) se nao verde
    texts = []
    for node, (x, y) in pos.items():
        if node in players:
            texts.append(plt.text(x, y, str(node), fontsize=8, color='blue'))
        else:
            texts.append(plt.text(x, y, node, fontsize=8, color='green'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.legend(['Players', 'Games'], loc='upper right')

    plt.title('Bipartite Graph of Players and Games')
    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.spring_layout(B, k=0.2)
    nx.draw_networkx_nodes(B, pos, nodelist=players, node_color='lightblue', node_size=50, label='Players')
    nx.draw_networkx_nodes(B, pos, nodelist=games, node_color='lightgreen', node_size=50, label='Games')
    nx.draw_networkx_edges(B, pos, alpha=0.5)
    texts = []
    for node, (x, y) in pos.items():
        if node in players:
            texts.append(plt.text(x, y, str(node), fontsize=8, color='blue'))
        else:
            texts.append(plt.text(x, y, node, fontsize=8, color='green'))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    plt.legend(['Players', 'Games'], loc='upper right')
    plt.title('Bipartite Graph of Players and Games')
    st.pyplot(fig)
#deixa a pagina stateful igual flutter
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = False

if st.sidebar.button('Show interactive dataframe'):
    st.session_state.sidebar_open = not st.session_state.sidebar_open

if st.session_state.sidebar_open:
    dif_option = {
    'Player_ID': 'multiselect', #selecionar por jogador é interessante
    'Game_title': 'multiselect', #selecionar por jogo é interessante
    'Behavior': 'multiselect', #apenas 2 opçoes portanto melhor
    }

    all_widgets = sp.create_widgets(df_filtred, dif_option)
    res = sp.filter_df(df_filtred, all_widgets)

    st.write(res)

