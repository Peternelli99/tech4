
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
    return joblib.load("models/gb_model (1).joblib")

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

    aba1, aba2, aba3, aba4, aba5 = st.tabs([
            "📊 Demografia", 
            "👨‍👩‍👧‍👦 Histórico Familiar", 
            "⚖️ Altura x Peso", 
            "🏃‍♂️ Atividade Física",
            "🍔 Comportamento Alimentar"
        ])

    with aba1:
        if df_filtrado.empty:
            st.warning("❌ Não existem registros para os filtros selecionados.")
        else:
            card1, card2, card3 = st.columns(3)
            card1.metric("Média de Idade", f"{df_filtrado['Age'].mean():.1f} anos")
            card2.metric("Total de Mulheres", len(df_filtrado[df_filtrado["Gender"] == "Female"]))
            card3.metric("Total de Homens", len(df_filtrado[df_filtrado["Gender"] == "Male"]))
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribuição de Obesidade por Gênero")
                df_temp1 = df_filtrado.copy()
                df_temp1["Obesity"] = df_temp1["Obesity"].map(rotulos["obesidade_tradutor"])
                df_temp1["Obesity"] = pd.Categorical(df_temp1["Obesity"], categories=ordem_obesidade, ordered=True)
                df_temp1["Gender"] = df_temp1["Gender"].map(rotulos["genero_tradutor"])
                if df_temp1[["Obesity", "Gender"]].dropna().empty:
                    st.info("🔍 Não existem registros suficientes para gerar este gráfico.")
                else:
                    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
                    pd.crosstab(df_temp1["Obesity"], df_temp1["Gender"]).loc[ordem_obesidade].plot(kind='bar', ax=ax1)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig1)
            
            with col2:
                st.subheader("Distribuição da Idade por Categoria de Obesidade")
                fig_age, ax_age = plt.subplots(figsize=(7, 4.5))
                sns.histplot(data=df_temp1, x="Age", hue="Obesity", multiple="stack", ax=ax_age)
                plt.tight_layout()
                st.pyplot(fig_age)
    
            
            with st.expander("📌 Ver Insight"):
                if df_filtrado[["Obesity", "Gender"]].dropna().empty:
                    st.info("📌 Não existem dados disponíveis para gerar insights.")
                else:
                    tabela_percent = pd.crosstab(df_filtrado["Obesity"], df_filtrado["Gender"], normalize='columns') * 100
                    st.dataframe(tabela_percent.round(1))
                    st.markdown("""
                    - **Mulheres** demonstram maior prevalência em **obesidade III**.
                    - **Homens** se concentram mais nas faixas de **obesidade II** e **sobrepeso**.
                    """)

    with aba2:
        
        col_fam1, col_fam2 = st.columns(2)
        with col_fam1:
            st.subheader("Obesidade por Histórico Familiar")
            df_temp2 = df_filtrado.copy()
            df_temp2["Obesity"] = df_temp2["Obesity"].map(rotulos["obesidade_tradutor"])
            df_temp2["Obesity"] = pd.Categorical(df_temp2["Obesity"], categories=ordem_obesidade, ordered=True)
            df_temp2["family_history"] = df_temp2["family_history"].map(rotulos["historico_tradutor"])
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            pd.crosstab(df_temp2["Obesity"], df_temp2["family_history"]).loc[ordem_obesidade].plot(kind='bar', ax=ax2)
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        with col_fam2:
            st.subheader("Peso vs historico familiar")
            fig_peso_hist, ax_peso_hist = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_filtrado, x="family_history", y="Weight", ax=ax_peso_hist)
            st.pyplot(fig_peso_hist)

       
        with st.expander("📌 Ver Insight"):
            st.markdown("""
            - Indivíduos com **histórico familiar positivo** apresentam maior frequência nos níveis de obesidade severa.
            - Esse fator pode indicar predisposição genética relevante.
            """)

    with aba3:
        card_hp1, card_hp2 = st.columns(2)
        imc_medio = (df_filtrado["Weight"] / df_filtrado["Height"]**2).mean()
        card_hp1.metric("IMC Médio", f"{imc_medio:.1f}")
        card_hp2.metric("Peso Médio", f"{df_filtrado['Weight'].mean():.1f} kg")
        
        col_hp1, col_hp2 = st.columns(2)
        with col_hp1:
            st.subheader("Altura vs Peso por Categoria")
            df_temp4 = df_filtrado.copy()
            df_temp4["Obesity"] = df_temp4["Obesity"].map(rotulos["obesidade_tradutor"])
            df_temp4["Obesity"] = pd.Categorical(df_temp4["Obesity"], categories=ordem_obesidade, ordered=True)
            fig4, ax4 = plt.subplots()
            sns.scatterplot(data=df_temp4, x="Height", y="Weight", hue="Obesity", ax=ax4)
            st.pyplot(fig4)

        with col_hp2:
            st.subheader("Relação de peso por Categoria de obesidade ")
            fig_boxpeso, ax_boxpeso = plt.subplots()
            sns.boxplot(data=df_temp4, x="Obesity", y="Weight", order=ordem_obesidade, ax=ax_boxpeso)
            plt.xticks(rotation=45)
            st.pyplot(fig_boxpeso)

        st.subheader("Relação de Altura por Categoria obesidade")
        fig_boxaltura, ax_boxaltura = plt.subplots()
        sns.boxplot(data=df_temp4, x="Obesity", y="Height", order=ordem_obesidade, ax=ax_boxaltura)
        plt.xticks(rotation=45)
        st.pyplot(fig_boxaltura)


        
        with st.expander("📌 Ver Insight"):
            st.markdown("""
            - A tendência é de que **maiores pesos para uma mesma altura** estejam associados a níveis mais graves de obesidade.
            - A visualização permite **identificar outliers** e zonas de risco.
            """)

    with aba4:

        card_faf1, card_faf2 = st.columns(2)
        card_faf1.metric("FAF médio", f"{df_filtrado['FAF'].mean():.2f}")
        card_faf2.metric("Mediana de FAF", f"{df_filtrado['FAF'].median():.2f}")
        
        col_faf_grafico, col_faf_insight = st.columns(2)
        with col_faf_grafico:
            st.subheader("Atividade Física por Categoria de Obesidade")
            df_temp4["Obesity"] = pd.Categorical(df_temp4["Obesity"], categories=ordem_obesidade, ordered=True)
            fig5, ax5 = plt.subplots()
            sns.boxplot(data=df_temp4, x="Obesity", y="FAF", order=ordem_obesidade, ax=ax5)
            plt.xticks(rotation=45)
            st.pyplot(fig5)

        with col_faf_insight:
            st.subheader("Distribuição do Tempo de Atividade Física por Nível de Obesidade")
            df_temp_faf = df_filtrado.copy()
            df_temp_faf["Obesity"] = df_temp_faf["Obesity"].map(rotulos["obesidade_tradutor"])
            df_temp_faf["Obesity"] = pd.Categorical(df_temp_faf["Obesity"], categories=ordem_obesidade, ordered=True)

            fig_faf_hist, ax_faf_hist = plt.subplots(figsize=(8, 4))
            sns.histplot(
                data=df_temp_faf,
                x="FAF",
                hue="Obesity",
                multiple="fill",  # melhora a visualização empilhando proporcionalmente
                palette="Set2",
                hue_order=ordem_obesidade,
                edgecolor="black",
                binwidth=0.25
            )
            ax_faf_hist.set_title("Distribuição do Tempo de Atividade Física por Nível de Obesidade")
            ax_faf_hist.set_xlabel("FAF (frequência de atividade física semanal)")
            ax_faf_hist.set_ylabel("Proporção")
            st.pyplot(fig_faf_hist)


    
        with st.expander("📌 Ver Insight"):
            st.markdown("""
            - Pacientes com **obesidade III** possuem, em média, menor atividade física semanal.
            - Os grupos com **peso normal** e **abaixo do peso** apresentam maior variabilidade na prática de atividades físicas.
            - A categoria **sobrepeso I** mostra comportamento semelhante ao grupo de obesidade I em termos de atividade física.
            """)

    with aba5:
        card_freq1, card_freq2 = st.columns(2)
        caec_freq = df_filtrado[df_filtrado["CAEC"] != "Nunca"]
        card_freq1.metric("Faz lanches fora de hora", f"{len(caec_freq)} pessoas")
        card_freq2.metric("Consome comida calórica", f"{len(df_filtrado[df_filtrado['FAVC'] == 'Sim'])} pessoas")

        df_temp5 = df_filtrado.copy()
        df_temp5["Obesity"] = df_temp5["Obesity"].map(rotulos["obesidade_tradutor"])
        df_temp5["Obesity"] = pd.Categorical(df_temp5["Obesity"], categories=ordem_obesidade, ordered=True)
        df_temp5["CAEC"] = df_temp5["CAEC"].map(rotulos["caec_tradutor"])
        df_temp5["FAVC"] = df_temp5["FAVC"].map(rotulos["favc_tradutor"])

        col_caec, col_favc = st.columns(2)

        with col_caec:
            st.subheader("Obesidade por Frequência de Lanches Fora de Hora")
            fig6, ax6 = plt.subplots()
            pd.crosstab(df_temp5["Obesity"], df_temp5["CAEC"]).loc[ordem_obesidade].plot(kind="bar", ax=ax6)
            plt.xticks(rotation=45)
            st.pyplot(fig6)

        with col_favc:
            st.subheader("Obesidade por Consumo de Comida Calórica")
            fig7, ax7 = plt.subplots()
            pd.crosstab(df_temp5["Obesity"], df_temp5["FAVC"]).loc[ordem_obesidade].plot(kind="bar", ax=ax7)
            plt.xticks(rotation=45)
            st.pyplot(fig7)

        with st.expander("📌 Ver Insight"):
            st.markdown("""
            - O **consumo frequente de lanches fora de hora** está correlacionado com maiores níveis de obesidade.
            - Indivíduos que **não consomem comida calórica** têm maior proporção nas categorias **peso normal** ou **abaixo do peso**.
            - A combinação dos dois comportamentos alimentares pode indicar **maior risco de obesidade severa**.
            """)


elif pagina == "Previsão Individual":
    st.title("Previsão Individual de Obesidade")
    st.markdown("Insira as informações para prever o nível de obesidade de um indivíduo.")
    # (A parte de previsão individual permanece inalterada)
