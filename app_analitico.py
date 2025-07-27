
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Carregar dados e modelo
@st.cache_data
def load_data():
    return pd.read_csv("data/Obesity.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/gb_model.joblib")

df = load_data()
model = load_model()

# Sidebar de navegação
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Painel Analítico", "Previsão Individual"])

if page == "Painel Analítico":
    st.title("Dashboard Analítico de Obesidade")
    st.markdown("Insights sobre perfil de obesidade com base no estudo")

    # Filtros otimizados
    st.sidebar.header("Filtros")
    genders = st.sidebar.multiselect("Gênero", df["Gender"].unique(), default=df["Gender"].unique())
    ages = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()),
                            (int(df["Age"].min()), int(df["Age"].max())))
    family = st.sidebar.multiselect("Histórico Familiar", df["family_history"].unique(), default=df["family_history"].unique())
    caec = st.sidebar.multiselect("Lanches fora de hora", df["CAEC"].unique(), default=df["CAEC"].unique())
    favc = st.sidebar.multiselect("Comida Calórica Frequente", df["FAVC"].unique(), default=df["FAVC"].unique())

    df_filt = df[
        (df["Gender"].isin(genders)) &
        (df["Age"].between(*ages)) &
        (df["family_history"].isin(family)) &
        (df["CAEC"].isin(caec)) &
        (df["FAVC"].isin(favc))
    ]

    # Métricas
    st.subheader("Visão Geral")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de registros", len(df_filt))
    col2.metric("Média de Peso (kg)", f"{df_filt['Weight'].mean():.1f}")
    col3.metric("Média de Altura (m)", f"{df_filt['Height'].mean():.2f}")

    # Distribuição de obesidade
    st.subheader("Distribuição de Obesidade")
    dist = df_filt["Obesity"].value_counts(normalize=True).mul(100)
    st.bar_chart(dist)

    # Obesidade por gênero
    st.subheader("Distribuição de Obesidade por Gênero")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    pd.crosstab(df_filt["Obesity"], df_filt["Gender"]).plot(kind='bar', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Obesidade por histórico familiar
    st.subheader("Obesidade vs Histórico Familiar")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    pd.crosstab(df_filt["Obesity"], df_filt["family_history"]).plot(kind='bar', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Peso por categoria
    st.subheader("Distribuição de Peso por Nível de Obesidade")
    fig3, ax3 = plt.subplots()
    df_filt.boxplot(column="Weight", by="Obesity", ax=ax3)
    plt.title("Peso por Categoria de Obesidade")
    plt.suptitle("")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Atividade física média por categoria
    st.subheader("Atividade Física Média por Categoria de Obesidade")
    media_faf = df_filt.groupby("Obesity")["FAF"].mean().sort_values()
    st.bar_chart(media_faf)

    # Refeições principais por categoria
    st.subheader("Média de Refeições por Categoria de Obesidade")
    media_ncp = df_filt.groupby("Obesity")["NCP"].mean().sort_values()
    st.bar_chart(media_ncp)

    # Texto final
    st.markdown("### 🩺 Insights para a equipe médica:")
    st.markdown("""
    - Peso e atividade física são bons indicadores para diferenciar os níveis de obesidade.
    - Há padrões de alimentação distintos entre os grupos (refeições principais e lanches).
    - Histórico familiar e comportamento alimentar devem ser considerados na triagem.
    """)

elif page == "Previsão Individual":
    st.title("Previsão Individual de Obesidade")

    st.sidebar.header("Nova previsão")
    with st.sidebar.form("predict_form"):
        inputs = {
            "Gender": st.selectbox("Gender", df["Gender"].unique()),
            "Age": st.number_input("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].mean())),
            "Height": st.number_input("Height", float(df["Height"].min()), float(df["Height"].max()), float(df["Height"].mean())),
            "Weight": st.number_input("Weight", float(df["Weight"].min()), float(df["Weight"].max()), float(df["Weight"].mean())),
            "family_history": st.selectbox("Histórico Familiar", df["family_history"].unique()),
            "FAVC": st.selectbox("Comida Calórica Freq.", df["FAVC"].unique()),
            "FCVC": st.slider("Freq. consumo vegetais", 1.0, 3.0, 2.0),
            "NCP": st.slider("Nº refeições principais", 1.0, 4.0, 3.0),
            "CAEC": st.selectbox("Lanches fora de hora", df["CAEC"].unique()),
            "SMOKE": st.selectbox("Fuma?", df["SMOKE"].unique()),
            "CH2O": st.slider("Copos de água por dia", 1.0, 3.0, 2.0),
            "SCC": st.selectbox("Monitora calorias?", df["SCC"].unique()),
            "FAF": st.slider("Atividade física semanal", 0.0, 3.0, 1.0),
            "TUE": st.slider("Horas em telas por dia", 0.0, 2.0, 1.0),
            "CALC": st.selectbox("Consumo de álcool", df["CALC"].unique()),
            "MTRANS": st.selectbox("Transporte mais usado", df["MTRANS"].unique())
        }

        submitted = st.form_submit_button("Prever Obesidade")
        if submitted:
            X_new = pd.DataFrame([inputs])
            pred = model.predict(X_new)[0]
            st.sidebar.success(f"Nível previsto: **{pred}**")
