
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Carregar dados e modelo
@st.cache_data
def carregar_dados():
    return pd.read_csv("data/Obesity.csv")

@st.cache_resource
def carregar_modelo():
    return joblib.load("models/gb_model.joblib")

df = carregar_dados()
modelo = carregar_modelo()

# Navegação lateral
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para:", ["Painel Analítico", "Previsão Individual"])

if pagina == "Painel Analítico":
    st.title("Painel Analítico de Obesidade")
    st.markdown("Análise de perfil de obesidade com base nos dados do estudo.")

    # Filtros
    st.sidebar.header("Filtros")
    generos = st.sidebar.multiselect("Gênero", df["Gender"].unique(), default=df["Gender"].unique())
    idades = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()),
                               (int(df["Age"].min()), int(df["Age"].max())))
    historico_familiar = st.sidebar.multiselect("Histórico Familiar de Obesidade", df["family_history"].unique(), default=df["family_history"].unique())
    lanches = st.sidebar.multiselect("Lanches Fora de Hora", df["CAEC"].unique(), default=df["CAEC"].unique())
    comida_calorica = st.sidebar.multiselect("Consumo Frequente de Comida Calórica", df["FAVC"].unique(), default=df["FAVC"].unique())

    df_filtrado = df[
        (df["Gender"].isin(generos)) &
        (df["Age"].between(*idades)) &
        (df["family_history"].isin(historico_familiar)) &
        (df["CAEC"].isin(lanches)) &
        (df["FAVC"].isin(comida_calorica))
    ]

    # Métricas
    st.subheader("Visão Geral")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros", len(df_filtrado))
    col2.metric("Média de Peso (kg)", f"{df_filtrado['Weight'].mean():.1f}")
    col3.metric("Média de Altura (m)", f"{df_filtrado['Height'].mean():.2f}")

    # Distribuição de obesidade
    st.subheader("Distribuição dos Níveis de Obesidade")
    dist = df_filtrado["Obesity"].value_counts(normalize=True).mul(100)
    st.bar_chart(dist)

    # Obesidade por gênero
    st.subheader("Distribuição de Obesidade por Gênero")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    pd.crosstab(df_filtrado["Obesity"], df_filtrado["Gender"]).plot(kind='bar', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Obesidade por histórico familiar
    st.subheader("Obesidade por Histórico Familiar")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    pd.crosstab(df_filtrado["Obesity"], df_filtrado["family_history"]).plot(kind='bar', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Peso por categoria
    st.subheader("Distribuição de Peso por Nível de Obesidade")
    fig3, ax3 = plt.subplots()
    df_filtrado.boxplot(column="Weight", by="Obesity", ax=ax3)
    plt.title("Peso por Categoria de Obesidade")
    plt.suptitle("")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Atividade física média
    st.subheader("Média de Atividade Física por Nível de Obesidade")
    media_faf = df_filtrado.groupby("Obesity")["FAF"].mean().sort_values()
    st.bar_chart(media_faf)

    # Número de refeições
    st.subheader("Média de Refeições por Nível de Obesidade")
    media_ncp = df_filtrado.groupby("Obesity")["NCP"].mean().sort_values()
    st.bar_chart(media_ncp)

    # Texto final
    st.markdown("### 🩺 Insights para a Equipe Médica:")
    st.markdown("""
    - Peso e atividade física são bons indicadores para diferenciar os níveis de obesidade.
    - Há padrões de alimentação distintos entre os grupos (refeições principais e lanches).
    - Histórico familiar e comportamento alimentar devem ser considerados na triagem.
    """)
