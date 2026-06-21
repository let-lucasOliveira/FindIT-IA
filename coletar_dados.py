"""
coletar_dados.py
Lê Candidatos, Questionario, Vagas e Align do Firestore
e salva em arquivos CSV locais para uso no treino do modelo.

Como usar:
    python coletar_dados.py

Pré-requisito:
    - Arquivo serviceAccountKey.json na mesma pasta
    - pip install firebase-admin pandas
"""

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os

# ── Inicializa o Firebase Admin ──────────────────────────────────────────────
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def coletar_colecao(nome_colecao):
    """Lê todos os documentos de uma coleção e retorna uma lista de dicts."""
    print(f"Coletando {nome_colecao}...")
    docs = db.collection(nome_colecao).stream()
    dados = []
    for doc in docs:
        d = doc.to_dict()
        d["_doc_id"] = doc.id  # guarda o ID do documento
        dados.append(d)
    print(f"  → {len(dados)} documentos encontrados")
    return dados

def main():
    os.makedirs("dados", exist_ok=True)

    # ── Candidatos ───────────────────────────────────────────────────────────
    candidatos = coletar_colecao("Candidatos")
    df_candidatos = pd.DataFrame(candidatos)
    df_candidatos.to_csv("dados/candidatos.csv", index=False)
    print(f"  Salvo em dados/candidatos.csv\n")

    # ── Questionario ─────────────────────────────────────────────────────────
    questionarios = coletar_colecao("Questionario")
    df_questionarios = pd.DataFrame(questionarios)
    df_questionarios.to_csv("dados/questionarios.csv", index=False)
    print(f"  Salvo em dados/questionarios.csv\n")

    # ── Vagas ────────────────────────────────────────────────────────────────
    vagas = coletar_colecao("Vagas")
    df_vagas = pd.DataFrame(vagas)
    df_vagas.to_csv("dados/vagas.csv", index=False)
    print(f"  Salvo em dados/vagas.csv\n")

    # ── Align ────────────────────────────────────────────────────────────────
    # Status reais do FindIT:
    #   "matched"        → match mútuo confirmado → label 1
    #   "rejeitado_cand" → candidato rejeitou      → label 0
    #   "rejeitado_emp"  → empresa rejeitou        → label 0
    #   "pendente_cand", "pendente_emp", "pendente" → ainda em aberto, ignorado
    align = coletar_colecao("Align")
    df_align = pd.DataFrame(align)

    # DEBUG temporário
    print("Valores únicos de status encontrados:")
    print(df_align["status"].value_counts())
    print()

    status_validos = ["matched", "rejeitado_cand", "rejeitado_emp"]
    df_align = df_align[df_align["status"].isin(status_validos)]
    df_align["label"] = df_align["status"].apply(lambda x: 1 if x == "matched" else 0)
    df_align.to_csv("dados/align.csv", index=False)
    print(f"  Salvo em dados/align.csv")
    print(f"  → {len(df_align[df_align['label']==1])} matched (aceito)")
    print(f"  → {len(df_align[df_align['label']==0])} rejeitado\n")

    print("✅ Coleta concluída. Arquivos salvos na pasta dados/")

if __name__ == "__main__":
    main()