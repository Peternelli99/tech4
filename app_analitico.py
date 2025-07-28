
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Carregar traduções
with open("rotulos_traduzidos.json", encoding="utf-8") as f:
    rotulos = json.load(f)

@st.cache_data
def carregar_dados():
    return pd.read_csv("data/Obesity.csv")

@st.cache_resource
def carregar_modelo():
    return joblib.load("models/gb_model.joblib")

df = carregar_dados()
modelo = carregar_modelo()

# Sidebar de navegação
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para:", ["Painel Analítico", "Previsão Individual"])

if pagina == "Painel Analítico":
    st.title("Painel Analítico de Obesidade")
    st.markdown("Análise de perfil de obesidade com base nos dados do estudo.")

    # Filtros
    st.sidebar.header("Filtros")

    genero_opcoes = list(rotulos["genero_tradutor"].values())
    genero_selecionado = st.sidebar.multiselect("Gênero", genero_opcoes, default=genero_opcoes)
    genero_valores = [k for k, v in rotulos["genero_tradutor"].items() if v in genero_selecionado]

    idade = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    altura = st.sidebar.slider("Altura (m)", float(df["Height"].min()), float(df["Height"].max()), (float(df["Height"].min()), float(df["Height"].max())))
    peso = st.sidebar.slider("Peso (kg)", float(df["Weight"].min()), float(df["Weight"].max()), (float(df["Weight"].min()), float(df["Weight"].max())))

    hist_opcoes = list(rotulos["historico_tradutor"].values())
    hist_selecionado = st.sidebar.multiselect("Histórico Familiar", hist_opcoes, default=hist_opcoes)
    hist_valores = [k for k, v in rotulos["historico_tradutor"].items() if v in hist_selecionado]

    caec_opcoes = list(rotulos["caec_tradutor"].values())
    caec_selecionado = st.sidebar.multiselect("Lanches Fora de Hora", caec_opcoes, default=caec_opcoes)
    caec_valores = [k for k, v in rotulos["caec_tradutor"].items() if v in caec_selecionado]

    favc_opcoes = list(rotulos["favc_tradutor"].values())
    favc_selecionado = st.sidebar.multiselect("Consumo de Comida Calórica", favc_opcoes, default=favc_opcoes)
    favc_valores = [k for k, v in rotulos["favc_tradutor"].items() if v in favc_selecionado]

    df_filtrado = df[
        (df["Gender"].isin(genero_valores)) &
        (df["Age"].between(*idade)) &
        (df["Height"].between(*altura)) &
        (df["Weight"].between(*peso)) &
        (df["family_history"].isin(hist_valores)) &
        (df["CAEC"].isin(caec_valores)) &
        (df["FAVC"].isin(favc_valores))
    ]

    # Visão geral
    st.subheader("Visão Geral")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros", len(df_filtrado))
    col2.metric("Média de Peso (kg)", f"{df_filtrado['Weight'].mean():.1f}")
    col3.metric("Média de Altura (m)", f"{df_filtrado['Height'].mean():.2f}")

    # Distribuição de Obesidade
    st.subheader("Distribuição dos Níveis de Obesidade")
    dist = df_filtrado["Obesity"].map(rotulos["obesidade_tradutor"]).value_counts(normalize=True).mul(100)
    st.bar_chart(dist)

    if not dist.empty:
        maior_categoria = dist.idxmax()
        percentual = dist.max()
        st.markdown(f"🔎 Categoria mais frequente: **{maior_categoria}** com **{percentual:.1f}%** dos registros filtrados.")

    # Obesidade por Gênero
    st.subheader("Distribuição de Obesidade por Gênero")
    df_temp1 = df_filtrado.copy()
    df_temp1["Obesity"] = df_temp1["Obesity"].map(rotulos["obesidade_tradutor"])
    df_temp1["Gender"] = df_temp1["Gender"].map(rotulos["genero_tradutor"])
    fig1, ax1 = plt.subplots()
    pd.crosstab(df_temp1["Obesity"], df_temp1["Gender"]).plot(kind='bar', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Histórico Familiar
    st.subheader("Obesidade por Histórico Familiar")
    df_temp2 = df_filtrado.copy()
    df_temp2["Obesity"] = df_temp2["Obesity"].map(rotulos["obesidade_tradutor"])
    df_temp2["family_history"] = df_temp2["family_history"].map(rotulos["historico_tradutor"])
    fig2, ax2 = plt.subplots()
    pd.crosstab(df_temp2["Obesity"], df_temp2["family_history"]).plot(kind='bar', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Dispersão Altura x Peso
    st.subheader("Altura vs Peso por Categoria")
    df_temp4 = df_filtrado.copy()
    df_temp4["Obesity"] = df_temp4["Obesity"].map(rotulos["obesidade_tradutor"])
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df_temp4, x="Height", y="Weight", hue="Obesity", ax=ax4)
    st.pyplot(fig4)
    st.markdown("📌 Este gráfico mostra a relação visual entre altura, peso e categorias de obesidade.")

    # FAF (atividade física)
    st.subheader("Atividade Física por Categoria de Obesidade")
    fig5, ax5 = plt.subplots()
    sns.boxplot(data=df_temp4, x="Obesity", y="FAF", ax=ax5)
    plt.xticks(rotation=45)
    st.pyplot(fig5)

    # Insights finais
    st.markdown("### 🩺 Insights para a Equipe Médica:")
    st.markdown("""
    - O nível de obesidade apresenta forte associação com peso, altura, histórico familiar e atividade física.
    - Comportamentos como tabagismo e consumo de álcool devem ser considerados em estratégias de prevenção.
    - Padrões alimentares (lanches e comidas calóricas) também impactam os resultados.
    """)

elif pagina == "Previsão Individual":
    st.title("Previsão Individual de Obesidade")
    st.markdown("Insira as informações para prever o nível de obesidade de um indivíduo.")
    # (A parte de previsão individual permanece inalterada)
