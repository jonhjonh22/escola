import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import numpy as np

# Configuração da página
st.set_page_config(page_title="Painel da Academia de Boxe", layout="wide")

# Carregar dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("academia_boxe.csv")
    df.columns = df.columns.str.strip()
    return df

df = carregar_dados()

# Título
st.title("Painel da Academia de Boxe")

# Métricas principais
media_idade = df["Idade"].mean()
media_calorias = df["Calorias_Sessao"].mean()
total_vitorias = df["Vitorias_Sparring"].sum()

st.markdown("## Resumo dos Alunos")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Média de Idade", f"{media_idade:.1f} anos")
with col2:
    st.metric("Média de Calorias por Sessão", f"{media_calorias:.0f} kcal")
with col3:
    st.metric("Total de Vitórias em Sparring", total_vitorias)

st.divider()

# Gráficos
with st.container():
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.markdown("### Distribuição por Nível")
        fig, ax1 = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df, x="Nivel", ax=ax1, palette="coolwarm")
        ax1.set_xlabel("Nível")
        ax1.set_ylabel("Qtd de Alunos")
        st.pyplot(fig)

    with col_graf2:
        st.markdown("### Frequência Semanal")
        fig, ax2 = plt.subplots(figsize=(4, 3))
        sns.histplot(df["Frequencia_Semanal"], bins=7, kde=False, color="green")
        ax2.set_xlabel("Sessões por Semana")
        ax2.set_ylabel("Qtd de Alunos")
        st.pyplot(fig)

with st.container():
    col_graf3, col_graf4 = st.columns(2)

    with col_graf3:
        st.markdown("### Calorias por Sessão")
        fig, ax3 = plt.subplots(figsize=(4, 3))
        sns.boxplot(data=df, y="Calorias_Sessao", x="Nivel", palette="pastel", ax=ax3)
        ax3.set_ylabel("Calorias")
        st.pyplot(fig)

    with col_graf4:
        st.markdown("### Vitórias x Derrotas")
        fig, ax4 = plt.subplots(figsize=(4, 3))
        sns.scatterplot(data=df, x="Vitorias_Sparring", y="Derrotas_Sparring", hue="Nivel", ax=ax4)
        ax4.set_xlabel("Vitórias")
        ax4.set_ylabel("Derrotas")
        st.pyplot(fig)

st.divider()

# Exportar CSV
st.markdown("## Exportar Dados")
csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Baixar CSV",
    data=csv,
    file_name="academia_boxe_export.csv",
    mime="text/csv",
)

st.divider()

# Análise Binomial: probabilidade de vitórias
st.markdown("## Probabilidade de Vitórias (Distribuição Binomial)")
p_vitoria = df["Vitorias_Sparring"].sum() / (df["Vitorias_Sparring"] + df["Derrotas_Sparring"]).sum()

col_a, col_b = st.columns(2)
with col_a:
    n = st.slider("Número de alunos simulados", min_value=5, max_value=50, value=10)
with col_b:
    k = st.slider("Número mínimo de vitórias desejadas", min_value=1, max_value=50, value=5)

if k > n:
    st.error("O número de vitórias desejadas não pode ser maior que o número de alunos.")
else:
    prob_soma = 1 - binom.cdf(k-1, n, p_vitoria)
    st.write(f"Com uma taxa observada de vitórias de {p_vitoria:.2%},")
    st.write(f"A probabilidade de pelo menos {k} vitórias em {n} alunos é **{prob_soma:.2%}**")

    probs_binom = [binom.pmf(i, n, p_vitoria) for i in range(n+1)]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(n+1), probs_binom, color=["gray" if i < k else "orange" for i in range(n+1)])
    ax.set_xlabel("Número de Vitórias")
    ax.set_ylabel("Probabilidade")
    ax.set_title("Distribuição Binomial")
    st.pyplot(fig)

st.divider()

# Análise Poisson: calorias por sessão
st.markdown("## Calorias por Sessão (Distribuição de Poisson)")
media_calorias = df["Calorias_Sessao"].mean()
poisson_n = st.slider("Número de calorias alvo", min_value=100, max_value=1200, value=600, step=50)

poisson_pmf = [poisson.pmf(k, media_calorias) for k in range(poisson_n, poisson_n+20)]
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(range(poisson_n, poisson_n+20), poisson_pmf, color="purple")
ax.set_xlabel("Calorias")
ax.set_ylabel("Probabilidade")
ax.set_title("Distribuição de Poisson")
st.pyplot(fig)
