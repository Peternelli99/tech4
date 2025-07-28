
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_csv("data/Obesity.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/gb_model.joblib")

df = load_data()
model = load_model()

label_translate = {
    'Insufficient_Weight': 'Abaixo do Peso',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso I',
    'Overweight_Level_II': 'Sobrepeso II',
    'Obesity_Type_I': 'Obesidade I',
    'Obesity_Type_II': 'Obesidade II',
    'Obesity_Type_III': 'Obesidade III',
    'Male': 'Masculino',
    'Female': 'Feminino',
    'yes': 'Sim',
    'no': 'Não',
    'Always': 'Sempre',
    'Frequently': 'Frequente',
    'Sometimes': 'Às vezes',
    'no': 'Nunca'
}

df["Obesity"] = df["Obesity"].replace(label_translate)
df["Gender"] = df["Gender"].replace(label_translate)
df["family_history"] = df["family_history"].replace(label_translate)
df["CAEC"] = df["CAEC"].replace(label_translate)
df["FAVC"] = df["FAVC"].replace(label_translate)

st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Painel Analítico", "Previsão Individual"])

if page == "Painel Analítico":
    st.title("Dashboard Analítico de Obesidade")

    st.sidebar.header("Filtros")
    genders = st.sidebar.multiselect("Gênero", df["Gender"].unique(), default=df["Gender"].unique())
    ages = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    heights = st.sidebar.slider("Altura (m)", float(df["Height"].min()), float(df["Height"].max()), (float(df["Height"].min()), float(df["Height"].max())))
    weights = st.sidebar.slider("Peso (kg)", float(df["Weight"].min()), float(df["Weight"].max()), (float(df["Weight"].min()), float(df["Weight"].max())))
    family = st.sidebar.multiselect("Histórico Familiar", df["family_history"].unique(), default=df["family_history"].unique())
    caec = st.sidebar.multiselect("Lanches Fora de Hora", df["CAEC"].unique(), default=df["CAEC"].unique())
    favc = st.sidebar.multiselect("Consumo de Comida Calórica", df["FAVC"].unique(), default=df["FAVC"].unique())

    df_filt = df[
        (df["Gender"].isin(genders)) &
        (df["Age"].between(*ages)) &
        (df["Height"].between(*heights)) &
        (df["Weight"].between(*weights)) &
        (df["family_history"].isin(family)) &
        (df["CAEC"].isin(caec)) &
        (df["FAVC"].isin(favc))
    ]

    st.subheader("Distribuição dos Níveis de Obesidade")
    dist = df_filt["Obesity"].value_counts(normalize=True).mul(100)
    st.bar_chart(dist)
    with st.expander("🔎 Ver análise interpretativa"):
        cat = dist.idxmax()
        perc = dist.max()
        st.markdown(f"- Categoria mais frequente: **{cat}** com **{perc:.1f}%** dos registros filtrados.  
- Sugere foco em estratégias de prevenção ou tratamento para essa categoria.")

    st.subheader("Distribuição de Obesidade por Gênero")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    pd.crosstab(df_filt["Obesity"], df_filt["Gender"]).plot(kind='bar', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    with st.expander("🔎 Ver análise interpretativa"):
        st.markdown("- Diferenças perceptíveis entre gêneros indicam possível influência comportamental ou biológica.")

    st.subheader("Obesidade por Histórico Familiar")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    pd.crosstab(df_filt["Obesity"], df_filt["family_history"]).plot(kind='bar', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    with st.expander("🔎 Ver análise interpretativa"):
        st.markdown("- Pessoas com histórico familiar têm maior incidência de obesidade.")

    st.subheader("Atividade Física por Categoria de Obesidade")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df_filt, x="Obesity", y="FAF", ax=ax4)
    plt.xticks(rotation=45)
    st.pyplot(fig4)
    with st.expander("🔎 Ver análise interpretativa"):
        st.markdown("- Indivíduos com **obesidade** tendem a praticar **menos atividade física** comparado aos com peso normal ou abaixo.")

    st.subheader("Altura vs Peso por Categoria")
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=df_filt, x="Height", y="Weight", hue="Obesity", ax=ax5)
    st.pyplot(fig5)
    with st.expander("🔎 Ver análise interpretativa"):
        st.markdown("- Relação clara entre altura e peso por categoria de obesidade.")

    st.subheader("Média de Refeições por Categoria de Obesidade")
    media_ncp = df_filt.groupby("Obesity")["NCP"].mean().sort_values()
    st.bar_chart(media_ncp)
    with st.expander("🔎 Ver análise interpretativa"):
        st.markdown("- Padrões alimentares também diferem entre as categorias.")
