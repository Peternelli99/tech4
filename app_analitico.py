
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

    ordem_obesidade = [
        "Abaixo do Peso", "Peso Normal", "Sobrepeso I",
        "Sobrepeso II", "Obesidade I", "Obesidade II", "Obesidade III"
    ]

    # Distribuição de Obesidade
    col_dist, col_insight1 = st.columns([3, 2])

    with col_dist:
        st.subheader("Distribuição dos Níveis de Obesidade")
        dist = df_filtrado["Obesity"].map(rotulos["obesidade_tradutor"])
        dist = pd.Categorical(dist, categories=ordem_obesidade, ordered=True)
        dist = pd.Series(dist).value_counts(normalize=True).reindex(ordem_obesidade).fillna(0).mul(100)
        st.bar_chart(dist)

    with col_insight1:
        with st.expander("📌 Ver Insight"):
            if not dist.empty:
                maior_categoria = dist.idxmax()
                percentual = dist.max()
                st.markdown(f"""
                - A categoria mais comum é **{maior_categoria}** com **{percentual:.1f}%** dos registros filtrados.
                - Isso pode indicar um grupo de risco predominante no público analisado.
                """)

    aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Demografia", 
    "👨‍👩‍👧‍👦 Histórico Familiar", 
    "⚖️ Altura x Peso", 
    "🏃‍♂️ Atividade Física"
    ])

    with aba1:
        col_gen, col_insight2 = st.columns([3, 2])
        with col_gen:
            st.subheader("Distribuição de Obesidade por Gênero")
            df_temp1 = df_filtrado.copy()
            df_temp1["Obesity"] = df_temp1["Obesity"].map(rotulos["obesidade_tradutor"])
            df_temp1["Obesity"] = pd.Categorical(df_temp1["Obesity"], categories=ordem_obesidade, ordered=True)
            df_temp1["Gender"] = df_temp1["Gender"].map(rotulos["genero_tradutor"])
            fig1, ax1 = plt.subplots()
            pd.crosstab(df_temp1["Obesity"], df_temp1["Gender"]).loc[ordem_obesidade].plot(kind='bar', ax=ax1)
            plt.xticks(rotation=45)
            st.pyplot(fig1)

            st.subheader("Distribuição da Idade por Categoria de Obesidade")
            fig_age, ax_age = plt.subplots()
            sns.histplot(data=df_temp1, x="Age", hue="Obesity", multiple="stack", ax=ax_age)
            st.pyplot(fig_age)

            corr_cols = ["Age", "Height", "Weight", "FAF"]
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(df_filtrado[corr_cols].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
            st.pyplot(fig_corr)

        with col_insight2:
            with st.expander("📌 Ver Insight"):
                st.markdown("""
                - **Mulheres** demonstram maior prevalência em **obesidade III**.
                - **Homens** se concentram mais nas faixas de **obesidade II** e **sobrepeso**.
                """)

    with aba2:
        col_fam, col_insight3 = st.columns([3, 2])
        with col_fam:
            st.subheader("Obesidade por Histórico Familiar")
            df_temp2 = df_filtrado.copy()
            df_temp2["Obesity"] = df_temp2["Obesity"].map(rotulos["obesidade_tradutor"])
            df_temp2["Obesity"] = pd.Categorical(df_temp2["Obesity"], categories=ordem_obesidade, ordered=True)
            df_temp2["family_history"] = df_temp2["family_history"].map(rotulos["historico_tradutor"])
            fig2, ax2 = plt.subplots()
            pd.crosstab(df_temp2["Obesity"], df_temp2["family_history"]).loc[ordem_obesidade].plot(kind='bar', ax=ax2)
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        with col_insight3:
            with st.expander("📌 Ver Insight"):
                st.markdown("""
                - Indivíduos com **histórico familiar positivo** apresentam maior frequência nos níveis de obesidade severa.
                - Esse fator pode indicar predisposição genética relevante.
                """)

    with aba3:
        col_hp, col_insight4 = st.columns([3, 2])
        with col_hp:
            st.subheader("Altura vs Peso por Categoria")
            df_temp4 = df_filtrado.copy()
            df_temp4["Obesity"] = df_temp4["Obesity"].map(rotulos["obesidade_tradutor"])
            df_temp4["Obesity"] = pd.Categorical(df_temp4["Obesity"], categories=ordem_obesidade, ordered=True)
            fig4, ax4 = plt.subplots()
            sns.scatterplot(data=df_temp4, x="Height", y="Weight", hue="Obesity", ax=ax4)
            st.pyplot(fig4)
        with col_insight4:
            with st.expander("📌 Ver Insight"):
                st.markdown("""
                - A tendência é de que **maiores pesos para uma mesma altura** estejam associados a níveis mais graves de obesidade.
                - A visualização permite **identificar outliers** e zonas de risco.
                """)

    with aba4:
        col_faf_grafico, col_faf_insight = st.columns([3, 2])
        with col_faf_grafico:
            st.subheader("Atividade Física por Categoria de Obesidade")
            df_temp4["Obesity"] = pd.Categorical(df_temp4["Obesity"], categories=ordem_obesidade, ordered=True)
            fig5, ax5 = plt.subplots()
            sns.boxplot(data=df_temp4, x="Obesity", y="FAF", order=ordem_obesidade, ax=ax5)
            plt.xticks(rotation=45)
            st.pyplot(fig5)
        with col_faf_insight:
            with st.expander("📌 Ver Insight"):
                st.markdown("""
                - Pacientes com **obesidade III** possuem, em média, menor atividade física semanal.
                - Os grupos com **peso normal** e **abaixo do peso** apresentam maior variabilidade na prática de atividades físicas.
                - A categoria **sobrepeso I** mostra comportamento semelhante ao grupo de obesidade I em termos de atividade física.
                """)


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
