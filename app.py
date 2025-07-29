import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Caching de recursos ---
@st.cache_data
def load_data(path="data/Obesity.csv") -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model_and_encoders(
    path_model="models/gb_model.joblib",
    path_encoder="models/label_encoder.joblib",
    path_features="models/feature_names.joblib"
):
    try:
        model = joblib.load(path_model)
        label_encoder = joblib.load(path_encoder)
        feature_names = joblib.load(path_features)
        return model, label_encoder, feature_names
    except Exception as e:
        st.error(f"❌ Erro ao carregar recursos: {e}")
        st.stop()

# Função para realizar o pre processamento 
def preprocess_data(raw_df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    df = raw_df.copy()
    df['Height'] = df['Height'] / 100
    bin_map = {
        'Gender': {'Female':0, 'Male':1},
        'family_history': {'no':0, 'yes':1},
        'FAVC': {'no':0, 'yes':1},
        'SMOKE': {'no':0, 'yes':1},
        'SCC': {'no':0, 'yes':1}
    }
    for col, m in bin_map.items():
        df[col] = df[col].map(m)
    df = pd.get_dummies(df, columns=['CAEC','CALC','MTRANS'])
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    return df[expected_cols]

# Tradutores de colunas
def decode_label(encoded, encoder):
    return encoder.inverse_transform([encoded])[0]

def translate_class(pt_label: str) -> str:
    m = {
        'Insufficient_Weight':'Peso Insuficiente',
        'Normal_Weight':'Peso Normal',
        'Overweight_Level_I':'Sobrepeso Nível I',
        'Overweight_Level_II':'Sobrepeso Nível II',
        'Obesity_Type_I':'Obesidade Tipo I',
        'Obesity_Type_II':'Obesidade Tipo II',
        'Obesity_Type_III':'Obesidade Tipo III'
    }
    return m.get(pt_label, pt_label)

# LAyout dos filtros
def sidebar_inputs():
    st.sidebar.title("Histórico Clínico do Paciente")
    st.sidebar.markdown("Anamnese, mapeamento de aspectos físicos e comportamentais:")
    st.sidebar.header("1. Dados Pessoais")
    genero = st.sidebar.selectbox("Gênero", ["Masculino","Feminino"])
    idade = st.sidebar.number_input("Idade", 1, 120, 30)
    altura = st.sidebar.number_input("Altura (cm)", 50, 250, 170)
    peso = st.sidebar.number_input("Peso (kg)", 20, 300, 70)
    st.sidebar.header("2. Estilo de Vida")
    fam = st.sidebar.radio("Histórico familiar de sobrepeso?", ["Sim","Não"])
    favc = st.sidebar.radio("Consome alimentos calóricos frequentemente?", ["Sim","Não"])
    smoke = st.sidebar.radio("Fumante?", ["Sim","Não"])
    calc = st.sidebar.selectbox("Qual frequencia consome alcool?", ["Não consome","Às vezes","Frequentemente","Sempre"])
    ncp = st.sidebar.slider("Quantas refeicoes ao dia?", 1, 4, 3)
    fcvc = st.sidebar.slider("Consome Vegetais?", 1, 3, 2)
    caec = st.sidebar.selectbox("Consome alimentos entre as refeicoes?", ["Não come","Às vezes","Frequentemente","Sempre"])
    ch2o = st.sidebar.slider("Qual consumo de agua diaria?", 1, 3, 2)
    faf = st.sidebar.slider("Realiza quantas vezes atividades fisica na semana?", 0, 7, 3)
    calorias = st.sidebar.radio("Monitora calorias?", ["Sim","Não"])
    tue = st.sidebar.selectbox("Quanto tempo utiliza dispositivos eletronicos?", ["Até 2h","Até 5h","Mais de 5h"])
    mtrans = st.sidebar.selectbox(
        "Utiliza qual tipo de transporte?",
        ["Automóvel","Motocicleta","Bicicleta","Transporte público","Caminhada"]
    )
    gender_map = {"Masculino":"Male", "Feminino":"Female"}
    yesno = {"Sim":"yes", "Não":"no"}
    freq = {"Não come":"no", "Não consome":"no", "Às vezes":"Sometimes", "Frequentemente":"Frequently", "Sempre":"Always"}
    trans = {"Automóvel":"Automobile", "Motocicleta":"Motorbike", "Bicicleta":"Bike", "Transporte público":"Public_Transportation", "Caminhada":"Walking"}
    tue_map = {"Até 2h":0, "Até 5h":1, "Mais de 5h":2}
    return {
        "Gender": gender_map[genero],
        "Age": idade,
        "Height": altura,
        "Weight": peso,
        "family_history": yesno[fam],
        "FAVC": yesno[favc],
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": freq[caec],
        "CH2O": ch2o,
        "SMOKE": yesno[smoke],
        "SCC": yesno[calorias],
        "FAF": faf,
        "TUE": tue_map[tue],
        "CALC": freq[calc],
        "MTRANS": trans[mtrans]
    }

# Processamento
df = load_data()
model, encoder, feature_names = load_model_and_encoders()
inputs = sidebar_inputs()

if st.sidebar.button("🔍 Analisar"):
    raw_df = pd.DataFrame([inputs])
    X = preprocess_data(raw_df, feature_names)
    if X.shape[1] != len(feature_names):
        st.error(f"❌ Esperado {len(feature_names)} features, recebeu {X.shape[1]}")
    else:
        num_pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        class_raw = decode_label(num_pred, encoder)
        class_pt = translate_class(class_raw)
        conf = max(proba) * 100
        st.subheader("🔎 Resultado da Análise")
        c1, c2 = st.columns(2)
        c1.success(class_pt)
        c2.info(f"{conf:.1f}%")

        # IMC
        altura_m = inputs['Height'] / 100
        bmi = inputs['Weight'] / altura_m ** 2
        st.subheader("🎯 Indice de massa corporal (IMC)")
        m1, m2 = st.columns(2)
        m1.metric("", f"{bmi:.1f}")
        status = (
            'Abaixo do peso' if bmi < 18.5 else
            'IMC Controlado' if bmi < 25 else
            'Sobrepeso' if bmi < 30 else
            'Obesidade'
        )
        m2.metric("", status)

        # Probabilidades
        st.subheader("📊 Probabilidades por Categoria")
        classes_pt = [translate_class(c) for c in encoder.classes_]
        dfp = pd.DataFrame({"Categoria": classes_pt, "Prob": proba * 100}).sort_values("Prob")
        fig, ax = plt.subplots()
        bars = ax.barh(dfp.Categoria, dfp.Prob)
        for i, b in enumerate(bars):
            if dfp.Categoria.iloc[i] == class_pt:
                b.set_color('red')
        ax.set_xlabel('Prob (%)')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)

        # Histograma
        st.subheader("📈 Distribuição de IMC na População")
        fig2, ax2 = plt.subplots()
        data_bmi = df.Weight / ((df.Height / 100) ** 2)
        ax2.hist(data_bmi, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(bmi, color='red', linestyle='--')
        st.pyplot(fig2)

        # Importância de features
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Importância das Features")
            fi = pd.Series(model.feature_importances_, index=feature_names).sort_values()
            fig3, ax3 = plt.subplots()
            bars3 = ax3.barh(fi.index, fi.values)
            for i, v in enumerate(fi.values):
                ax3.text(v + 0.001, i, f"{v:.3f}")
            ax3.grid(axis='x', alpha=0.3)
            st.pyplot(fig3)

        # Insights Personalizados
            st.subheader("📊 Insights Personalizados ")
            ins = []
            if inputs['family_history'] == 'yes':
                ins.append("Histórico familiar de sobrepeso: maior predisposição genética.")
            if inputs['FAVC'] == 'yes':
                ins.append("Consumo frequente de alimentos calóricos: avalie reduzir calorias vazias.")
            if inputs['SMOKE'] == 'yes':
                ins.append("Tabagismo pode influenciar metabolismo; considere parar.")
            if inputs['CALC'] in ['Frequently', 'Always']:
                ins.append("Alto consumo de álcool: modere para apoiar controle de peso.")
            if inputs['CH2O'] < 2:
                ins.append("Consumo de água abaixo do ideal: objetive 2-3 litros/dia.")
            else:
                ins.append("Hidratação adequada.")
            for item in ins:
                st.markdown(f"- {item}")

        # Resumo de Preenchimento
        st.markdown("### 📋 Resumo de Preenchimento")
        order = [
            'Gender','Age','Height','Weight','family_history',
            'FAVC','NCP','FCVC','CAEC','CH2O','SMOKE','SCC','FAF','TUE','CALC','MTRANS'
        ]
        display_names = [
            'Gênero','Idade','Altura (cm)','Peso (kg)','Histórico Familiar',
            'Consome alimentos calóricos frequentemente?','NCP','FCVC','Consome alimentos entre as refeicoes?','Qual consumo de agua diaria?','Fumante?','SCC','Realiza quantas vezes atividades fisica na semana?','TUE','Monitora calorias?','Utiliza qual tipo de transporte?'
        ]
        resumo_vals = [inputs[k] for k in order]
        df_summary = pd.DataFrame({
            'Item': display_names,
            'Valor': resumo_vals
        })
        st.table(df_summary)

# Sobre o Modelo
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Modelo**: Gradient Boosting  \n"
    f"**Features**: {len(feature_names)}  \n"
    f"**Classes**: {len(encoder.classes_)}  \n"
    f"**Acurácia**: 94.8%"
)
