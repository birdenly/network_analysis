import networkx as nx
import streamlit as st
import pandas as pd
import seaborn as sns
import streamlit_pandas as sp
from adjustText import adjust_text
import numpy as np
import powerlaw
from networkx.algorithms import community

import matplotlib.pyplot as plt

def close_interaction():
    st.session_state.sidebar_open = False

@st.cache_data
def load_data():
    df = pd.read_csv(file, header=None, names=["Player_ID", "Game_title", "Behavior", "Hours", "Ignore"], nrows=2500,dtype={'Player_ID': str})
    return df
@st.cache_data #Apenas para mostrar antes/depois
def unfiltred_load_data():
    df = pd.read_csv(file, header=None, names=["Player_ID", "Game_title", "Behavior", "Hours", "Ignore"],dtype={'Player_ID': str})
    return df

file = "./steam-200k.csv"
df = load_data()
df_un = unfiltred_load_data()
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

if st.sidebar.button('Show main concepts about network analysis'):
    close_interaction()

    st.write('--------')
    st.write("Choose a Tab below to show one concept about network analysis using the Steam dataset") 
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13= st.tabs([
    "Adjacency Matrix",
    "Diameter",
    "Periphery",
    "Empirical Degree Distribution Histogram",
    "Local Clustering Coefficient for Selected Nodes",
    "Global Clustering Coefficient",
    "Strongly and Weakly Connected Components",
    "Degree Centrality",
    "Closeness Centrality",
    "Betweenness Centrality",
    "Eigenvector Centrality",
    "General Network Assortativity",
    "Communities"
])

    with tab1:
        st.header("Adjacency matrix")
        st.write("Clicando em cada coluna podemos ver Jogos comprados ou jogados por cada jogador, ou se clicarmos em um jogo podemos ver aqueles que compraram ou jogaram o jogo")
        matrix = nx.adjacency_matrix(B).todense()
        nodes = list(B.nodes)
        df = pd.DataFrame(matrix, index=nodes, columns=nodes)
        st.write(df)

    with tab2:
        st.header("Diameter")
        st.write("O diametro de um grafo é o maior caminho dos caminhos mais curto entre dois nós no graph")
        st.write('Por ser um graph bipartido é dificil ele ser conectado, portanto não se pode achar o diametro dele.')
        st.write('Caso networkx.diameter(B) seja usado, ira retornar o Erro: Found infinite path length because the graph is not connected')
        st.write('Por isso, iremos calcular o diametro de cada componente conectado')
        st.write("-------------------")
        connected_components = list(nx.connected_components(B))

        max_diameter = []
        for component in connected_components:
            subgraph = B.subgraph(component)
            if len(subgraph) > 1:
                diameter = nx.diameter(subgraph)
                max_diameter.append(diameter)

        for i, diameter in enumerate(max_diameter, 1):
            st.write("Componectado conectado ", i)
            st.write("Diametro: ", diameter)
            st.write("-------------------")

    with tab3:
        st.header("Periphery")
        st.write("A periferia de um graph é o subgrafo de nós que tem a maior excentricidade")
        st.write('Por ser um graph bipartido é dificil ele ser conectado, portanto não se pode achar a periferia dele.')
        st.write('Caso networkx.periphery(B) seja usado, ira retornar o Erro: Found infinite path length because the graph is not connected')
        st.write('Por isso, iremos calcular a periferia de cada componente conectado')
        st.write("-------------------")
        connected_components = list(nx.connected_components(B))

        peripheries = []
        for component in connected_components:
            subgraph = B.subgraph(component)
            if len(subgraph) > 1:
                periphery = nx.periphery(subgraph)
                peripheries.append(periphery)

        for i, periphery in enumerate(peripheries, 1):
            st.write("Componectado conectado ", i)
            st.write("Periferia: ", periphery)
            st.write("-------------------")

    with tab4:
        degrees = dict(nx.degree(B))
        st.header("Empirical Degree Distribution Histogram")
        st.write("Alta quantidade de elementos que se conectam pouco a outros. Provavelmente em sua maioria jogos de nicho ou pessoas que jogam pouco")
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.histplot(degrees, bins=30, kde=True)
        plt.title('Histograma de distribuição empírica de grau')
        plt.xlabel('degree')
        plt.ylabel('Frequencia')
        st.pyplot(fig)

        degree_values = np.array(list(degrees.values()))

        fit = powerlaw.Fit(degree_values, discrete=True)

        st.header("Empirical Degree Distribution with power law")
        st.write("https://youtu.be/imY4ap0bg1U?t=118")
        st.write("De acordo com o video acima a probabilidade de um no ter grau k é proporcional a k^-alpha. Onde **alpha**é o expoente da lei de potencia.")
        st.write("No video é mostrado que **alpha** tende a ser 2 < **alpha** <  3. **alpha** sendo maior que 2 significa que maior parte dos nos tem poucas conexões e alpha sendo menor que 3 significa que existe nos nos tem muitas conexões (HUBS)")
        st.write("O **alpha** da Distribuição Empírica de Grau com Lei de Potência da steam é 1.87, então o **alpha** apenas atende a condição de **alpha** < 3, portanto existe muitos nos com muitas conexões (HUBS)")
        fig, ax = plt.subplots(figsize=(15, 10))
        powerlaw.plot_pdf(degree_values, ax=ax, color='b', marker='o', label='Empirical Data')
        fit.power_law.plot_pdf(ax=ax, color='r', linestyle='--', label=f'Alpha={fit.alpha:.2f}')
        plt.title('Histograma de Distribuição Empírica de Grau com Lei de Potência')
        plt.xlabel('degree')
        plt.ylabel('Frequência')
        plt.legend()
        st.pyplot(fig)

        st.write(f"Expoente (alpha): {fit.alpha:.2f}")

    with tab5:
        st.header("Local Clustering Coefficient for the 35 highest players")
        st.write('O coeficiente de clustering local é mede o grau com que o nó de um grafo tende a agrupar-se')
        st.write('É necessario criar novo grafico, utilizando a função weighted_projected_graph, de players que serão conectados caso se liguem ao mesmo jogo, necessario já que graficos bipartidos normalmente tem clustering 0')
        st.write("Maior clustering = joga jogos mais populares")
        player_projection = nx.bipartite.weighted_projected_graph(B, players)

        clustering_coeffs = nx.clustering(player_projection, weight='weight')

        sorted_coeffs = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)
        selected_nodes = [node for node, _ in sorted_coeffs[:35]]
        selected_clustering_coeffs = {node: clustering_coeffs[node] for node in selected_nodes}

        df_clustering = pd.DataFrame(list(selected_clustering_coeffs.items()), columns=['nos', 'clustering'])

        fig, ax = plt.subplots(figsize=(15, 10))
        sns.barplot(x='nos', y='clustering', data=df_clustering)
        plt.title('Coeficiente de clustering para os 35 jogadores com maior coeficiente')
        plt.xlabel('Nos')
        plt.ylabel('Coeficiente de clustering')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tab6:
        st.header("Global Clustering Coefficient")
        st.write('O coeficiente de clustering global é mede o grau com que os nós de um grafo tendem a agrupar-se')
        st.write('É necessario criar um novo grafico, utilizando a função weighted_projected_graph, de players que serão conectados caso se liguem ao mesmo jogo, necessario já que graficos bipartidos normalmente tem clustering 0')
        st.write("Maior clustering = joga jogos mais populares")
        player_projection = nx.bipartite.weighted_projected_graph(B, players)

        global_clustering = nx.transitivity(player_projection)

        st.write(f"Coeficiente de clustering global: {global_clustering}")
        st.write("-------------------")
        
        clustering_coefficients = nx.clustering(player_projection)
        for player, coefficient in clustering_coefficients.items():
            st.write("Player: ",player)
            st.write("Clustering Coefficient: ",coefficient)
            st.write("-------------------")

    with tab7:
        st.header("Strongly and Weakly Connected Components")
        st.write("Nem o graph ou seus sub grafos são direcionados, portanto não se pode achar os componentes fortemente conectados dele.")
        st.write("networkx.is_directed(G)")
        st.write("networkx.is_directed(player_projection)")
        st.write("networkx.is_directed(B)")
        st.write("networkx.is_strongly_connected(G)  ERROR: not implemented for undirected type")
        st.write("networkx.is_weakly_connected(G) ERROR: not implemented for undirected type")
    with tab8:
        st.header("Degree Centrality")
        st.write('Degree centrality é o numero de conexões que um nó tem.')
        st.write('Degree centrality de todos os nos, apenas de jogadores e apenas de jogos:')

        degree_centrality = nx.degree_centrality(B)

        df_degree = pd.DataFrame(list(degree_centrality.items()), columns=['Node', 'degree'])

        plt.figure(figsize=(12, 8))
        sns.histplot(df_degree['degree'], bins=30, kde=True, color='purple')
        plt.title('degree Distribution for All Nodes')
        plt.xlabel('degree')
        plt.ylabel('frequencia')
        st.pyplot(plt)

        player_centrality = {node: centrality for node, centrality in degree_centrality.items() if node in players} #apenas players
        game_centrality = {node: centrality for node, centrality in degree_centrality.items() if node in games} #apenas games

        df_player_centrality = pd.DataFrame(list(player_centrality.items()), columns=['Player_ID', 'Degree Centrality'])
        df_game_centrality = pd.DataFrame(list(game_centrality.items()), columns=['Game_title', 'Degree Centrality'])

        plt.figure(figsize=(12, 8))
        sns.histplot(df_player_centrality['Degree Centrality'], bins=30, kde=True)
        plt.title('Degree Centrality for Players')
        plt.xlabel('Degree Centrality')
        plt.ylabel('frequencia')
        plt.show()
        st.pyplot(plt)

        plt.figure(figsize=(12, 8))
        sns.histplot(df_game_centrality['Degree Centrality'], bins=30, kde=True)
        plt.title('Degree Centrality for Games')
        plt.xlabel('Degree Centrality')
        plt.ylabel('frequencia')
        plt.show()
        st.pyplot(plt)
    with tab9:
        st.header("Closeness Centrality")
        st.write("Closeness Centrality é a média da menor distância de um nó para todos os outros nós.")
        st.write("Closeness Centrality de todos os nos, apenas de jogadores e apenas de jogos:")
        closeness_centrality = nx.closeness_centrality(B)

        df_closeness = pd.DataFrame(list(closeness_centrality.items()), columns=['Node', 'Closeness Centrality'])

        plt.figure(figsize=(12, 8))
        sns.histplot(df_closeness['Closeness Centrality'], bins=30, kde=True, color='purple')
        plt.title('Closeness Centrality Distribution for All Nodes')
        plt.xlabel('Closeness Centrality')
        plt.ylabel('frequencia')
        st.pyplot(plt)

        player_centrality = {node: centrality for node, centrality in closeness_centrality.items() if node in players} #apenas players
        game_centrality = {node: centrality for node, centrality in closeness_centrality.items() if node in games} #apenas games

        df_player_centrality = pd.DataFrame(list(player_centrality.items()), columns=['Player_ID', 'Closeness Centrality'])
        df_game_centrality = pd.DataFrame(list(game_centrality.items()), columns=['Game_title', 'Closeness Centrality'])

        plt.figure(figsize=(12, 8))
        sns.histplot(df_player_centrality['Closeness Centrality'], bins=30, kde=True)
        plt.title('Closeness Centrality for Players')
        plt.xlabel('Closeness Centrality')
        plt.ylabel('frequencia')
        plt.show()
        st.pyplot(plt)

        plt.figure(figsize=(12, 8))
        sns.histplot(df_game_centrality['Closeness Centrality'], bins=30, kde=True)
        plt.title('Closeness Centrality for Games')
        plt.xlabel('Closeness Centrality')
        plt.ylabel('frequencia')
        plt.show()
        st.pyplot(plt)
    with tab10:
        st.header("Betweenness Centrality")
        st.write("Betweenness Centrality é a quantidade de vezes que o nó é o mais curto caminho entre dois outros nós.")
        betweenness_centrality = nx.betweenness_centrality(B)

        df_between = pd.DataFrame(list(betweenness_centrality.items()), columns=['Node', 'Betweenness Centrality'])

        plt.figure(figsize=(12, 8))
        sns.histplot(df_between['Betweenness Centrality'], bins=30, kde=True, color='purple')
        plt.title('Betweenness Centralityy Distribution for All Nodes')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('frequencia')
        st.pyplot(plt)

        
        player_centrality = {node: centrality for node, centrality in betweenness_centrality.items() if node in players} #apenas players
        game_centrality = {node: centrality for node, centrality in betweenness_centrality.items() if node in games} #apenas games

        df_player_centrality = pd.DataFrame(list(player_centrality.items()), columns=['Player_ID', 'Betweenness Centrality'])
        df_game_centrality = pd.DataFrame(list(game_centrality.items()), columns=['Game_title', 'Betweenness Centrality'])

        plt.figure(figsize=(12, 8))
        sns.histplot(df_player_centrality['Betweenness Centrality'], bins=30, kde=True)
        plt.title('Betweenness Centrality for Players')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('frequencia')
        plt.show()
        st.pyplot(plt)

        plt.figure(figsize=(12, 8))
        sns.histplot(df_game_centrality['Betweenness Centrality'], bins=30, kde=True)
        plt.title('Betweenness Centrality for Games')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('frequencia')
        plt.show()
        st.pyplot(plt)
    with tab11:
        st.header("Eigenvector Centrality")
        st.write("Eigenvector Centrality é a medida de quão influente um nó é em uma rede.")
        Eigenvector_Centrality = nx.eigenvector_centrality(B,max_iter=2500, tol=1e-08)

        df_between = pd.DataFrame(list(Eigenvector_Centrality.items()), columns=['Node', 'Eigenvector Centrality'])

        plt.figure(figsize=(12, 8))
        sns.histplot(df_between['Eigenvector Centrality'], bins=30, kde=True, color='purple')
        plt.title('Eigenvector Centrality Distribution for All Nodes')
        plt.xlabel('Eigenvector Centrality')
        plt.ylabel('frequencia')
        st.pyplot(plt)

        player_centrality = {node: centrality for node, centrality in Eigenvector_Centrality.items() if node in players} #apenas players
        game_centrality = {node: centrality for node, centrality in Eigenvector_Centrality.items() if node in games} #apenas games

        df_player_centrality = pd.DataFrame(list(player_centrality.items()), columns=['Player_ID', 'Eigenvector Centrality'])
        df_game_centrality = pd.DataFrame(list(game_centrality.items()), columns=['Game_title', 'Eigenvector Centrality'])

        plt.figure(figsize=(12, 8))
        sns.histplot(df_player_centrality['Eigenvector Centrality'], bins=30, kde=True)
        plt.title('Eigenvector Centrality for Players')
        plt.xlabel('Eigenvector Centrality')
        plt.ylabel('frequencia')
        plt.show()
        st.pyplot(plt)

        plt.figure(figsize=(12, 8))
        sns.histplot(df_game_centrality['Eigenvector Centrality'], bins=30, kde=True)
        plt.title('Eigenvector Centrality for Games')
        plt.xlabel('Eigenvector Centrality')
        plt.ylabel('frequencia')
        plt.show()
        st.pyplot(plt)
    with tab12:
        st.header("General Network Assortativity")
        st.write("A Assortatividade de um grafo é uma medida de tendência de nós com graus semelhantes se conectarem uns aos outros.")
        st.write("A Assortatividade vai de -1 a 1. No caso do grafo da steam ela é -0.4, que significa que a rede é disassortativa, ou seja, nós com graus diferentes tendem a se conectar.")
        degree_assortativity = nx.degree_assortativity_coefficient(B)
        st.write("Assortatividade geral da rede: ",degree_assortativity)
    with tab13:
        st.header("Communities")
        st.write("Comunidades são subgrafos densamente conectados em um grafo.")
        st.write("Inves de analisar a comunidade com todos os nos (jogadores e jogos juntos), é necessario criar um novo grafico, utilizando a função weighted_projected_graph, de jogadores que serão conectados caso se liguem ao mesmo jogo")
        player_projection = nx.bipartite.weighted_projected_graph(B, players)

        communities = community.greedy_modularity_communities(player_projection)
        player_projection = nx.bipartite.weighted_projected_graph(B, players)
        communities = community.greedy_modularity_communities(player_projection)

        plt.figure(figsize=(10, 10))

        color_map = []
        for node in player_projection:
            for i, community in enumerate(communities):
                if node in community:
                    color_map.append(i)
                    break
        pos = nx.kamada_kawai_layout(player_projection)
        nx.draw(player_projection, pos, node_color=color_map,with_labels=True)
        st.pyplot(plt)

if st.sidebar.button('Show unfiltred dataset'):
    close_interaction()
    
    st.write(df_un)

if st.sidebar.button('Information about the columns'):
    close_interaction()

    p, g, b , h= st.columns(4)
    p.write(df_filtred['Player_ID'].describe())
    g.write(df_filtred['Game_title'].describe())
    b.write(df_filtred['Behavior'].describe())
    h.write(df_filtred['Hours'].describe())


if st.sidebar.button('Show a sample of the complete graph'):
    close_interaction()

    steam_sorted = df.sort_values(by=['Behavior'], ascending=False)

    filtered_steam = steam_sorted.drop_duplicates(subset=['Player_ID', 'Game_title'])

    filtered_steam_sample = filtered_steam.sample(n=200)

    G = nx.Graph()

    players = filtered_steam_sample['Player_ID'].unique()
    games = filtered_steam_sample['Game_title'].unique()

    B.add_nodes_from(players, bipartite=0)
    B.add_nodes_from(games, bipartite=1)

    for index, row in filtered_steam_sample.iterrows():
        G.add_edge(row['Player_ID'], row['Game_title'])

    plt.figure(figsize=(20, 15))

    plt.title('Graph of Players and Games (spring_layout)')
    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.2)
    nx.draw_networkx_nodes(G, pos, nodelist=players, node_color='lightblue', node_size=50, label='Players')
    nx.draw_networkx_nodes(G, pos, nodelist=games, node_color='lightgreen', node_size=50, label='Games')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
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


    plt.figure(figsize=(20, 15))
    plt.title('Graph of Players and Games (bipartite_layout)')

    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.bipartite_layout(G, players,scale=2, center=(0,0))

    nx.draw_networkx_nodes(G, pos, nodelist=players, node_color='lightblue', node_size=50, label='Players')
    nx.draw_networkx_nodes(G, pos, nodelist=games, node_color='lightgreen', node_size=50, label='Games')
    nx.draw_networkx_edges(G, pos, alpha=0.5)

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

#manter a pagina aberta
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

