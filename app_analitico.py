
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json

# Carregar traduções
with open("rotulos_traduzidos.json", encoding="utf-8") as f:
    rotulos = json.load(f)

# Carregar dados e modelo
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

    # Filtros com tradução
    st.sidebar.header("Filtros")

    genero_opcoes = list(rotulos["genero_tradutor"].values())
    genero_selecionado = st.sidebar.multiselect("Gênero", genero_opcoes, default=genero_opcoes)
    genero_valores = [k for k, v in rotulos["genero_tradutor"].items() if v in genero_selecionado]

    idade = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

    hist_opcoes = list(rotulos["historico_tradutor"].values())
    hist_selecionado = st.sidebar.multiselect("Histórico Familiar de Obesidade", hist_opcoes, default=hist_opcoes)
    hist_valores = [k for k, v in rotulos["historico_tradutor"].items() if v in hist_selecionado]

    caec_opcoes = list(rotulos["caec_tradutor"].values())
    caec_selecionado = st.sidebar.multiselect("Lanches Fora de Hora", caec_opcoes, default=caec_opcoes)
    caec_valores = [k for k, v in rotulos["caec_tradutor"].items() if v in caec_selecionado]

    favc_opcoes = list(rotulos["favc_tradutor"].values())
    favc_selecionado = st.sidebar.multiselect("Consumo Frequente de Comida Calórica", favc_opcoes, default=favc_opcoes)
    favc_valores = [k for k, v in rotulos["favc_tradutor"].items() if v in favc_selecionado]

    df_filtrado = df[
        (df["Gender"].isin(genero_valores)) &
        (df["Age"].between(*idade)) &
        (df["family_history"].isin(hist_valores)) &
        (df["CAEC"].isin(caec_valores)) &
        (df["FAVC"].isin(favc_valores))
    ]

    # Métricas
    st.subheader("Visão Geral")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros", len(df_filtrado))
    col2.metric("Média de Peso (kg)", f"{df_filtrado['Weight'].mean():.1f}")
    col3.metric("Média de Altura (m)", f"{df_filtrado['Height'].mean():.2f}")

    # Gráficos traduzidos
    st.subheader("Distribuição dos Níveis de Obesidade")
    dist = df_filtrado["Obesity"].map(rotulos["obesidade_tradutor"]).value_counts(normalize=True).mul(100)
    st.bar_chart(dist)

    st.subheader("Distribuição de Obesidade por Gênero")
    df_temp1 = df_filtrado.copy()
    df_temp1["Obesity"] = df_temp1["Obesity"].map(rotulos["obesidade_tradutor"])
    df_temp1["Gender"] = df_temp1["Gender"].map(rotulos["genero_tradutor"])
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    pd.crosstab(df_temp1["Obesity"], df_temp1["Gender"]).plot(kind='bar', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("Obesidade por Histórico Familiar")
    df_temp2 = df_filtrado.copy()
    df_temp2["Obesity"] = df_temp2["Obesity"].map(rotulos["obesidade_tradutor"])
    df_temp2["family_history"] = df_temp2["family_history"].map(rotulos["historico_tradutor"])
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    pd.crosstab(df_temp2["Obesity"], df_temp2["family_history"]).plot(kind='bar', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("Distribuição de Peso por Nível de Obesidade")
    df_temp3 = df_filtrado.copy()
    df_temp3["Obesity"] = df_temp3["Obesity"].map(rotulos["obesidade_tradutor"])
    fig3, ax3 = plt.subplots()
    df_temp3.boxplot(column="Weight", by="Obesity", ax=ax3)
    plt.title("Peso por Categoria de Obesidade")
    plt.suptitle("")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.subheader("Média de Atividade Física por Nível de Obesidade")
    media_faf = df_temp3.groupby("Obesity")["FAF"].mean().sort_values()
    st.bar_chart(media_faf)

    st.subheader("Média de Refeições por Nível de Obesidade")
    media_ncp = df_temp3.groupby("Obesity")["NCP"].mean().sort_values()
    st.bar_chart(media_ncp)

    st.markdown("### 🩺 Insights para a Equipe Médica:")
    st.markdown("""
    - Peso e atividade física são bons indicadores para diferenciar os níveis de obesidade.
    - Há padrões de alimentação distintos entre os grupos (refeições principais e lanches).
    - Histórico familiar e comportamento alimentar devem ser considerados na triagem.
    """)
