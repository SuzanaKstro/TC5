{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPbeqjVk4cEk0xwWnEn8DtV",
      "include_colab_link":True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/SuzanaKstro/TC5/blob/main/trabalhofinal.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "z8kmBAMpt3XB"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import joblib\n",
        "import xgboost as xgb\n",
        "\n",
        "# Carrega o modelo treinado\n",
        "modelo = joblib.load('modelo_xgb.pkl')\n",
        "\n",
        "# Define os campos que o modelo espera e seus nomes amigáveis\n",
        "campos = {\n",
        "    'perfil_vaga.estado': 'Estado da vaga',\n",
        "    'perfil_vaga.cidade': 'Cidade da vaga',\n",
        "    'perfil_vaga.regiao': 'Região da vaga',\n",
        "    'perfil_vaga.nivel_academico': 'Nível acadêmico da vaga',\n",
        "    'perfil_vaga.nivel_ingles': 'Nível de inglês da vaga',\n",
        "    'perfil_vaga.nivel_espanhol': 'Nível de espanhol da vaga',\n",
        "    'perfil_vaga.faixa_etaria': 'Faixa etária da vaga',\n",
        "    'perfil_vaga.horario_trabalho': 'Horário de trabalho',\n",
        "    'perfil_vaga.areas_atuacao': 'Área de atuação da vaga',\n",
        "    'perfil_vaga.vaga_especifica_para_pcd': 'Vaga específica para PCD?',\n",
        "    'informacoes_basicas.tipo_contratacao': 'Tipo de contratação',\n",
        "    'informacoes_basicas.prioridade_vaga': 'Prioridade da vaga',\n",
        "    'informacoes_profissionais.area_atuacao': 'Área de atuação do candidato',\n",
        "    'informacoes_profissionais.nivel_profissional': 'Nível profissional do candidato',\n",
        "    'formacao_e_idiomas.nivel_academico': 'Nível acadêmico do candidato',\n",
        "    'formacao_e_idiomas.nivel_ingles': 'Nível de inglês do candidato',\n",
        "    'formacao_e_idiomas.nivel_espanhol': 'Nível de espanhol do candidato',\n",
        "    'formacao_e_idiomas.outro_idioma': 'Outro idioma',\n",
        "    'formacao_e_idiomas.instituicao_ensino_superior': 'Instituição de ensino superior',\n",
        "    'formacao_e_idiomas.ano_conclusao': 'Ano de conclusão da formação'\n",
        "}\n",
        "\n",
        "# Título\n",
        "st.title('🧠 Previsão de Contratação - Decision AI')\n",
        "\n",
        "# Inputs do usuário\n",
        "st.markdown(\"### Preencha os dados da vaga e do candidato\")\n",
        "entrada = {}\n",
        "\n",
        "for campo, nome_amigavel in campos.items():\n",
        "    if campo == 'perfil_vaga.vaga_especifica_para_pcd':\n",
        "        valor = st.selectbox(nome_amigavel, ['Não', 'Sim'])\n",
        "        entrada[campo] = True if valor == 'Sim' else False\n",
        "    elif campo == 'formacao_e_idiomas.ano_conclusao':\n",
        "        entrada[campo] = st.number_input(nome_amigavel, min_value=1950, max_value=2050, step=1)\n",
        "    else:\n",
        "        entrada[campo] = st.text_input(nome_amigavel)\n",
        "\n",
        "# Botão para prever\n",
        "if st.button('Fazer Previsão'):\n",
        "    df_input = pd.DataFrame([entrada])\n",
        "\n",
        "    # Codificação one-hot\n",
        "    df_input = pd.get_dummies(df_input)\n",
        "\n",
        "    # Garante que todas as colunas esperadas pelo modelo estejam presentes\n",
        "    colunas_esperadas = modelo.get_booster().feature_names\n",
        "    colunas_faltantes = [col for col in colunas_esperadas if col not in df_input.columns]\n",
        "\n",
        "    # Cria colunas faltantes com valor 0\n",
        "    df_faltantes = pd.DataFrame(0, index=[0], columns=colunas_faltantes)\n",
        "    df_input = pd.concat([df_input, df_faltantes], axis=1)\n",
        "\n",
        "    # Reordena as colunas conforme esperado pelo modelo\n",
        "    df_input = df_input[colunas_esperadas]\n",
        "\n",
        "    # Previsão\n",
        "    proba = modelo.predict_proba(df_input)[0][1]\n",
        "    classe = modelo.predict(df_input)[0]\n",
        "\n",
        "    st.markdown(f\"### Resultado:\")\n",
        "    st.write(f\"**Classe prevista:** {'Contratado' if classe == 1 else 'Não contratado'}\")\n",
        "    st.write(f\"**Probabilidade de contratação:** {proba:.2%}\")\n",
        "    st.balloons()  # Animação de balões para celebrar a previsão\n",
        "    st.success(\"Previsão realizada com sucesso!\")\n",
        "\n",
        "\n"
      ]
    }
  ]
}
