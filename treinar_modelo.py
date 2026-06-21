"""
treinar_modelo.py (v2 — adaptado ao schema do DebugSyncActivity)
Lê os CSVs gerados pelo coletar_dados.py, faz o pré-processamento,
treina o LambdaMART com divisão 70/30 e salva o modelo.

Mudanças nesta versão:
  - Vaga agora usa "area" (Mobile/Backend/Dados/Frontend/DevOps/Gestão) e
    "nivelExperiencia" ("Sênior"/"Pleno"/"Junior") — sem stack/setor/modalidade.
  - "salario" é numérico (Double), não string formatada.
  - Questionario foi recriado por SeedCandidatosQuestionario, alinhado
    com essas mesmas áreas.

Como usar:
    python treinar_modelo.py

Pré-requisito:
    - Ter rodado coletar_dados.py antes
    - pip install pandas scikit-learn lightgbm rapidfuzz
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import ast
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz

# ── Carrega os CSVs ──────────────────────────────────────────────────────────
print("Carregando dados...")
df_candidatos    = pd.read_csv("dados/candidatos.csv")
df_questionarios = pd.read_csv("dados/questionarios.csv")
df_vagas         = pd.read_csv("dados/vagas.csv")
df_align         = pd.read_csv("dados/align.csv")

print(f"  {len(df_candidatos)} candidatos")
print(f"  {len(df_questionarios)} questionários")
print(f"  {len(df_vagas)} vagas")
print(f"  {len(df_align)} interações (matched + rejeitado)\n")

# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_list(valor):
    """Converte string de lista do CSV de volta pra lista Python."""
    if pd.isna(valor):
        return []
    if isinstance(valor, list):
        return valor
    try:
        return ast.literal_eval(valor)
    except:
        return [str(valor)]

def similaridade_texto(a, b):
    """Ratio + Token Sort + Partial Ratio — pega o maior score (0-1)."""
    if not a or not b:
        return 0.0
    a, b = str(a).lower().strip(), str(b).lower().strip()
    score = max(
        fuzz.ratio(a, b),
        fuzz.token_sort_ratio(a, b),
        fuzz.partial_ratio(a, b)
    )
    return round(score / 100, 4)

def match_lista(lista_candidato, texto_vaga):
    """Proporção de itens da lista do candidato que aparecem no texto da vaga."""
    if not lista_candidato or not texto_vaga:
        return 0.0
    lista = safe_list(lista_candidato)
    texto = str(texto_vaga).lower()
    if not lista:
        return 0.0
    matches = sum(1 for item in lista if str(item).lower() in texto)
    return round(matches / len(lista), 4)

def nivel_para_numero(nivel):
    """Converte nível textual em número — cobre 'Sênior'/'sênior'/'senior' etc."""
    mapa = {
        "entrando na área": 0, "estagio": 0, "estágio": 0, "estágio": 0,
        "junior": 1, "júnior": 1,
        "pleno": 2,
        "senior": 3, "sênior": 3,
    }
    return mapa.get(str(nivel).lower().strip(), 1)

# ── Monta o dataset de treino ────────────────────────────────────────────────
print("Montando dataset de treino...")

linhas = []

for _, align in df_align.iterrows():
    candidato_id = align.get("fkCandidatoId") or align.get("candidatoId")
    vaga_id      = align.get("fkVagaId")      or align.get("vagaId")
    label        = align["label"]

    # Busca candidato
    cand = df_candidatos[df_candidatos["id"] == candidato_id]
    if cand.empty:
        cand = df_candidatos[df_candidatos["_doc_id"] == candidato_id]
    if cand.empty:
        continue
    cand = cand.iloc[0]

    # Busca questionário
    quest = df_questionarios[df_questionarios["uid"] == candidato_id]
    if quest.empty:
        quest = df_questionarios[df_questionarios["_doc_id"] == candidato_id]
    if quest.empty:
        # sem questionário não dá pra montar features de intenção — pula
        continue
    quest = quest.iloc[0]

    # Busca vaga (schema novo: id pode estar em "id" custom ou _doc_id)
    vaga = df_vagas[df_vagas["_doc_id"] == vaga_id]
    if vaga.empty and "id" in df_vagas.columns:
        vaga = df_vagas[df_vagas["id"] == vaga_id]
    if vaga.empty:
        continue
    vaga = vaga.iloc[0]

    # ── Features ─────────────────────────────────────────────────────────────

    # 1. Área desejada (questionário) vs área da vaga (campo "area")
    area_cand = quest.get("areaDesejada", "")
    area_vaga = vaga.get("area", vaga.get("titulo", ""))
    f_area = similaridade_texto(area_cand, area_vaga)

    # 2. Stack do candidato vs stack exigida na vaga (campo direto)
    stack_cand = quest.get("stackInteresse", "")
    stack_vaga = vaga.get("stack", "")
    f_stack = match_lista(stack_cand, stack_vaga)

    # 3. Setor de interesse vs setor da vaga (campo direto)
    setor_cand = quest.get("setorInteresse", "")
    setor_vaga = vaga.get("setor", "")
    f_setor = match_lista(setor_cand, setor_vaga)

    # 4. Nível atual do candidato vs nível exigido pela vaga
    nivel_cand = nivel_para_numero(quest.get("nivelAtual", "junior"))
    nivel_vaga = nivel_para_numero(vaga.get("nivelExperiencia", "Pleno"))
    f_nivel = max(0.0, 1.0 - abs(nivel_cand - nivel_vaga) / 3)

    # 5. Modalidade preferida vs modalidade da vaga (campo direto)
    modal_cand = str(quest.get("modalidade", "")).lower().strip()
    modal_vaga = str(vaga.get("modalidade", "")).lower().strip()
    f_modalidade = 1.0 if modal_cand == modal_vaga else 0.0

    # 6. Cidade do candidato vs cidade da vaga
    cidade_cand = str(cand.get("cidade", "")).lower()
    cidade_vaga = str(vaga.get("cidade", "")).lower()
    f_cidade = similaridade_texto(cidade_cand, cidade_vaga) 

    # 7. Pretensão salarial (candidato, número) vs salário da vaga (número)
    try:
        sal_cand = float(cand.get("pretensaoSalarial", 0) or 0)
        sal_vaga = float(vaga.get("salario", 0) or 0)
        if sal_vaga > 0 and sal_cand > 0:
            f_salario = min(1.0, sal_vaga / sal_cand)
        else:
            f_salario = 0.5
    except:
        f_salario = 0.5

    linhas.append({
        "candidato_id": candidato_id,
        "vaga_id":      vaga_id,
        "f_area":       f_area,
        "f_stack":      f_stack,
        "f_setor":      f_setor,
        "f_nivel":      f_nivel,
        "f_modalidade": f_modalidade,
        "f_cidade":     f_cidade,
        "f_salario":    f_salario,
        "label":        label,
    })

df_treino = pd.DataFrame(linhas)
print(f"  {len(df_treino)} linhas no dataset\n")

if len(df_treino) < 10:
    print("⚠️  Poucos dados de interação válidos (com questionário) para treinar.")
    print("   Confirme se SeedCandidatosQuestionario rodou e se Align referencia")
    print("   os mesmos candidato_id/vaga_id presentes em Candidatos/Vagas.")
    exit()

# ── Divisão 70/30 ────────────────────────────────────────────────────────────
print("Dividindo 70% treino / 30% teste...")

features = ["f_area", "f_stack", "f_setor", "f_nivel", "f_modalidade", "f_cidade", "f_salario"]

X = df_treino[features]
y = df_treino["label"]

# stratify só funciona se houver pelo menos 2 classes com >=2 amostras cada
estratificar = y.nunique() > 1 and y.value_counts().min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y if estratificar else None
)

print(f"  Treino: {len(X_train)} amostras")
print(f"  Teste:  {len(X_test)} amostras\n")

if not estratificar:
    print("⚠️  Atenção: o dataset não tem as duas classes (matched/rejeitado) "
          "em quantidade suficiente. O modelo pode não aprender a discriminar bem.\n")

# ── Treina o LambdaMART ──────────────────────────────────────────────────────
print("Treinando LambdaMART...")

groups_train = df_treino.loc[X_train.index].groupby("candidato_id").size().values
groups_test  = df_treino.loc[X_test.index].groupby("candidato_id").size().values

train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
test_data  = lgb.Dataset(X_test,  label=y_test,  group=groups_test, reference=train_data)

params = {
    "objective":    "lambdarank",
    "metric":       "ndcg",
    "ndcg_eval_at": [3, 5],
    "num_leaves":   31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 5,   # menor que o padrão (20) — datasets de TCC são pequenos
    "verbose":      -1,
}

callbacks = [lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=10)]

modelo = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[test_data],
    callbacks=callbacks,
)

# ── Salva o modelo ───────────────────────────────────────────────────────────
joblib.dump(modelo, "modelo_findit.pkl")
print("\n✅ Modelo salvo em modelo_findit.pkl")

# ── Importância das features ─────────────────────────────────────────────────
print("\nImportância de cada feature no modelo:")
importancias = dict(zip(features, modelo.feature_importance()))
max_imp = max(importancias.values()) or 1
for feat, imp in sorted(importancias.items(), key=lambda x: x[1], reverse=True):
    barra = "█" * int(imp / max_imp * 20)
    print(f"  {feat:<15} {barra} {imp}")