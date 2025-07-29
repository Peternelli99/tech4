
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json


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

def plot_crosstab(ax, df, row, col, ordem, title=""):
    crosstab = pd.crosstab(df[row], df[col])
    if crosstab.empty:
        st.info(f"🔍 Não há dados suficientes para o gráfico: {title}")
        return
    crosstab = crosstab.reindex(ordem, fill_value=0)
    crosstab.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)


st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para:", ["Painel Analítico"])

if pagina == "Painel Analítico":
    st.title("Painel Analítico de Obesidade")
    st.markdown("Análise de perfil de obesidade com base nos dados do estudo.")

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


    st.subheader("Visão Geral")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros", len(df_filtrado))
    col2.metric("Média de Peso (kg)", f"{df_filtrado['Weight'].mean():.1f}")
    col3.metric("Média de Altura (m)", f"{df_filtrado['Height'].mean():.2f}")

    ordem_obesidade = [
        "Abaixo do Peso", "Peso Normal", "Sobrepeso I",
        "Sobrepeso II", "Obesidade I", "Obesidade II", "Obesidade III"
    ]

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
                    crosstab = pd.crosstab(df_temp1["Obesity"], df_temp1["Gender"])
                    crosstab = crosstab.reindex(ordem_obesidade, fill_value=0)
                    crosstab.plot(kind='bar', ax=ax1)
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
            
                    total_fem = (df_filtrado["Gender"] == "Female").sum()
                    total_masc = (df_filtrado["Gender"] == "Male").sum()
            
                    if total_fem > 0 and total_masc == 0:
                        st.markdown("""
                        ### 🔍 Mulheres:
                        - Maior prevalência em **obesidade III**, indicando risco elevado.
                        - Há também concentração significativa nas faixas de **sobrepeso II** e **obesidade I**.
                        - A distribuição etária mostra que **a maioria está entre 20 e 25 anos**, com casos graves até os 40+.
                        """)
                    elif total_masc > 0 and total_fem == 0:
                        st.markdown("""
                        ### 🔍 Homens:
                        - A maior incidência está em **obesidade II** e **sobrepeso I/II**.
                        - Homens com **peso normal ou abaixo do peso** são menos comuns, indicando tendência ao excesso de peso.
                        - Idade majoritária entre **18 e 28 anos**, mas também há obesidade severa acima dos 30.
                        """)
                    else:
                        st.markdown("""
                        ### 🔍 Geral:
                        - **Homens** concentram-se em **obesidade II** e **sobrepeso**, enquanto **mulheres** apresentam maior número em **obesidade III**.
                        - Há uma distribuição consistente de obesidade moderada em ambos os sexos.
                        - A faixa etária predominante é entre **20 e 25 anos**, indicando uma população jovem já em níveis de obesidade preocupantes.
                        """)




    with aba2:
        
        card_fam1, card_fam2 = st.columns(2)
        hist_sim = df_filtrado[df_filtrado["family_history"] == "yes"]
        card_fam1.metric("Com histórico familiar", f"{len(hist_sim)} registros")
        card_fam2.metric("Sem histórico", f"{len(df_filtrado) - len(hist_sim)} registros")

        col_fam1, col_fam2 = st.columns(2)
        with col_fam1:
            st.subheader("Obesidade por Histórico Familiar")
            df_temp2 = df_filtrado.copy()
            df_temp2["Obesity"] = df_temp2["Obesity"].map(rotulos["obesidade_tradutor"])
            df_temp2["Obesity"] = pd.Categorical(df_temp2["Obesity"], categories=ordem_obesidade, ordered=True)
            df_temp2["family_history"] = df_temp2["family_history"].map(rotulos["historico_tradutor"])
            fig2, ax2 = plt.subplots(figsize=(6, 4.2))
            crosstab2 = pd.crosstab(df_temp2["Obesity"], df_temp2["family_history"])
            crosstab2 = crosstab2.reindex(ordem_obesidade, fill_value=0)
            crosstab2.plot(kind='bar', ax=ax2)
            plt.xticks(rotation=45)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig2)

        with col_fam2:
            st.subheader("Peso vs historico familiar")
            fig_peso_hist, ax_peso_hist = plt.subplots(figsize=(6, 4.2))
            sns.boxplot(data=df_filtrado, x="family_history", y="Weight", ax=ax_peso_hist)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig_peso_hist)

       
        with st.expander("📌 Ver Insight"):
            count_sim = df_filtrado[df_filtrado["family_history"] == "yes"].shape[0]
            count_nao = df_filtrado[df_filtrado["family_history"] == "no"].shape[0]

            if count_sim > 0 and count_nao == 0:
                st.markdown("""
                ### ✅ Apenas com histórico familiar
                - Indivíduos com **histórico familiar positivo** apresentam grande incidência de **obesidade tipo II e III**.
                - Praticamente não há registros de **peso normal ou insuficiente** nesse grupo.
                - A mediana de peso é **significativamente mais alta**, com presença de **outliers de peso elevado**.
                - Isso pode indicar uma **predisposição genética relevante**.
                """)

            elif count_nao > 0 and count_sim == 0:
                st.markdown("""
                ### 🚫 Apenas sem histórico familiar
                - Indivíduos **sem histórico familiar** concentram-se em **peso normal ou sobrepeso I**.
                - A distribuição de obesidade severa (tipos II e III) é praticamente inexistente.
                - O peso tende a ser **mais baixo e estável**, com **menor variabilidade**.
                - Isso sugere que **a ausência de predisposição genética pode ser um fator protetivo**.
                """)

            elif count_sim > 0 and count_nao > 0:
                st.markdown("""
                ### 🧬 Comparativo Geral: com vs sem histórico
                - Indivíduos com **histórico familiar** de obesidade têm **maior propensão** a níveis severos de obesidade.
                - A média e mediana de peso são **notavelmente maiores** nesse grupo.
                - Já os sem histórico se concentram mais em **faixas saudáveis**, com maior percentual de **peso normal**.
                - A **disparidade entre os grupos** reforça a hipótese de que **genética e ambiente familiar** influenciam fortemente o quadro de obesidade.
                """)

            else:
                st.info("📌 Não existem dados disponíveis para gerar insights.")



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
            fig4, ax4 = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(data=df_temp4, x="Height", y="Weight", hue="Obesity", ax=ax4)
            plt.tight_layout()
            st.pyplot(fig4)

        with col_hp2:
            st.subheader("Relação de peso por Categoria de obesidade ")
            fig_boxpeso, ax_boxpeso = plt.subplots(figsize=(7, 4.5))
            sns.boxplot(data=df_temp4, x="Obesity", y="Weight", order=ordem_obesidade, ax=ax_boxpeso)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_boxpeso)

        st.subheader("Relação de Altura por Categoria obesidade")
        fig_boxaltura, ax_boxaltura = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(data=df_temp4, x="Obesity", y="Height", order=ordem_obesidade, ax=ax_boxaltura)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_boxaltura)


        
        with st.expander("📌 Ver Insight"):
            if df_filtrado[["Height", "Weight", "Obesity"]].dropna().empty:
                st.info("📌 Não existem dados disponíveis para gerar insights.")
            else:
                imc_medio = df_filtrado["Weight"] / (df_filtrado["Height"] ** 2)
                imc_medio = imc_medio.mean()

                st.markdown(f"""
                ### 📏 Análise de Altura e Peso (dados Gerais)
                
                - O **IMC médio** da amostra é de aproximadamente **{imc_medio:.1f}**, o que indica **sobrepeso** segundo a classificação da OMS.
                - A relação entre **altura e peso** mostra que **quanto maior o peso para uma mesma altura**, mais provável é a associação com **obesidade severa**.
                - As categorias de **obesidade II e III** apresentam indivíduos com **altos pesos**, independentemente da altura, e muitos estão fora dos limites interquartis (outliers).
                - Já os grupos de **peso normal e abaixo do peso** tendem a se concentrar em alturas médias com pesos significativamente menores.
                - A análise da **altura isolada** por categoria de obesidade mostra uma **leve tendência de maior altura** nos grupos com obesidade moderada, mas sem grandes diferenças significativas.
                """)

                st.caption("ℹ️ Os gráficos ajudam a identificar padrões extremos (outliers) e comportamentos típicos por categoria de obesidade.")


    with aba4:

        pct_sedentarios = (df_filtrado["FAF"] == 0).mean() * 100

        card_faf1, card_faf2 = st.columns(2)
        card_faf1.metric("Sedentários", f"{pct_sedentarios:.1f}%", "FAF = 0")

        pct_ativos = (df_filtrado["FAF"] >= 2).mean() * 100
        card_faf2.metric("Fisicamente Ativos", f"{pct_ativos:.1f}%", "FAF ≥ 2")

        
        col_faf_grafico, col_faf_insight = st.columns(2)
        with col_faf_grafico:
            st.subheader("Atividade Física por Categoria de Obesidade")
            df_temp4["Obesity"] = pd.Categorical(df_temp4["Obesity"], categories=ordem_obesidade, ordered=True)
            fig5, ax5 = plt.subplots(figsize=(7, 4.5))
            sns.boxplot(data=df_temp4, x="Obesity", y="FAF", order=ordem_obesidade, ax=ax5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig5)

        with col_faf_insight:
            st.subheader("Distribuição do Tempo de Atividade Física por Nível de Obesidade")
            df_temp_faf = df_filtrado.copy()
            df_temp_faf["Obesity"] = df_temp_faf["Obesity"].map(rotulos["obesidade_tradutor"])
            df_temp_faf["Obesity"] = pd.Categorical(df_temp_faf["Obesity"], categories=ordem_obesidade, ordered=True)

            fig_faf_hist, ax_faf_hist = plt.subplots(figsize=(7, 4.5))
            sns.histplot(
                data=df_temp_faf,
                x="FAF",
                hue="Obesity",
                multiple="fill",
                palette="Set2",
                hue_order=ordem_obesidade,
                edgecolor="black",
                binwidth=0.25
            )
            ax_faf_hist.set_title("Distribuição do Tempo de Atividade Física por Nível de Obesidade")
            ax_faf_hist.set_xlabel("FAF (frequência de atividade física semanal)")
            ax_faf_hist.set_ylabel("Proporção")
            plt.tight_layout()
            st.pyplot(fig_faf_hist)


    
        with st.expander("📌 Ver Insight"):
            if df_filtrado["FAF"].dropna().empty:
                st.info("📌 Não existem dados disponíveis para gerar insights.")
            else:
                faf_mean = df_filtrado["FAF"].mean()
                faf_median = df_filtrado["FAF"].median()

                st.markdown(f"""
                ### 🏃 Análise de Atividade Física

                - O valor **médio** da frequência de atividade física semanal (FAF) é **{faf_mean:.2f}**, enquanto a **mediana** é **{faf_median:.2f}** — indicando uma **distribuição assimétrica**, com muitas pessoas relatando níveis baixos de atividade.
                - **Indivíduos com obesidade severa (tipo II e III)** tendem a praticar **menos atividade física** em comparação com os grupos de peso normal ou abaixo do peso.
                - O gráfico de proporção revela que, mesmo entre aqueles com **alta frequência de exercícios (FAF = 2 ou 3)**, ainda existem casos de **sobrepeso e obesidade**, o que pode indicar influência de **outros fatores como alimentação ou genética**.
                - Já os grupos com **FAF = 0** apresentam alta concentração de **obesidade tipo III**, reforçando a **associação entre sedentarismo e obesidade grave**.
                
                """)
                st.caption("ℹ️ FAF representa a frequência de atividade física semanal (escala de 0 a 3).")


    with aba5:
        card_freq1, card_freq2, card_freq3 = st.columns(3)
        caec_freq = df_filtrado[df_filtrado["CAEC"] != "no"]
        card_freq1.metric("Faz lanches fora de hora", f"{len(caec_freq)} pessoas")
        card_freq2.metric("Consome comida calórica", f"{len(df_filtrado[df_filtrado['FAVC'] == 'yes'])} pessoas")
        card_freq3.metric("Não consome comida calórica", f"{len(df_filtrado[df_filtrado['FAVC'] == 'no'])} pessoas")

        df_temp5 = df_filtrado.copy()
        df_temp5["Obesity"] = df_temp5["Obesity"].map(rotulos["obesidade_tradutor"])
        df_temp5["Obesity"] = pd.Categorical(df_temp5["Obesity"], categories=ordem_obesidade, ordered=True)
        df_temp5["CAEC"] = df_temp5["CAEC"].map(rotulos["caec_tradutor"])
        df_temp5["FAVC"] = df_temp5["FAVC"].map(rotulos["favc_tradutor"])

        col_caec, col_favc = st.columns(2)

        with col_caec:
            st.subheader("Obesidade por Frequência de Lanches Fora de Hora")
            fig6, ax6 = plt.subplots()
            crosstab5 = pd.crosstab(df_temp5["Obesity"], df_temp5["CAEC"])
            crosstab5 = crosstab5.reindex(ordem_obesidade, fill_value=0)
            crosstab5.plot(kind="bar", ax=ax6)
            plt.xticks(rotation=45)
            st.pyplot(fig6)

        with col_favc:
            st.subheader("Obesidade por Consumo de Comida Calórica")
            fig7, ax7 = plt.subplots()
            crosstab_favc = pd.crosstab(df_temp5["Obesity"], df_temp5["FAVC"])
            crosstab_favc = crosstab_favc.reindex(ordem_obesidade, fill_value=0)
            crosstab_favc.plot(kind="bar", ax=ax7)
            plt.xticks(rotation=45)
            st.pyplot(fig7)

        with st.expander("📌 Ver Insight"):
            st.markdown("""
            - **Frequência alta de lanches fora de hora** (principalmente “Às vezes”, “Frequentemente” e “Sempre”) está fortemente associada a maiores níveis de obesidade, especialmente do tipo II e III.
            - O **consumo de comida calórica** (FAVC = Sim) é predominante nas categorias de sobrepeso e obesidade — praticamente todos os casos graves de obesidade pertencem a esse grupo.
            - Indivíduos que **não consomem comida calórica** apresentam maior proporção de “Peso Normal” ou “Abaixo do Peso”, e são minoria nas categorias de obesidade.
            - A **combinação de ambos os comportamentos** (lanches fora de hora + consumo de comida calórica) marca o grupo de maior risco, com altíssimos números em obesidade severa.
            - Estratégias de prevenção devem focar na **redução do consumo de lanches entre as refeições** e no **controle da qualidade dos alimentos**.
            """)

