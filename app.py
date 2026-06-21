# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import numpy as np
import math
import ast
import os
import json
import base64
from rapidfuzz import fuzz

app = FastAPI()

# ── 1. Inicializa o Firebase Admin ──
# Em produção (Render): lê as credenciais da variável de ambiente FIREBASE_CREDENTIALS
# Em desenvolvimento local: usa o arquivo serviceAccountKey.json
_firebase_env = os.environ.get("FIREBASE_CREDENTIALS")
if _firebase_env:
    # Render armazena como JSON em base64 ou JSON puro
    try:
        _cred_dict = json.loads(base64.b64decode(_firebase_env).decode("utf-8"))
    except Exception:
        _cred_dict = json.loads(_firebase_env)
    cred = credentials.Certificate(_cred_dict)
else:
    cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred)
db = firestore.client()

# ── 2. Carrega o modelo LightGBM ──
modelo = joblib.load("modelo_findit.pkl")

# ── 3. Funções Helpers ──
def safe_list(valor):
    if not valor or str(valor) == "nan":
        return []
    if isinstance(valor, list):
        return valor
    try:
        return ast.literal_eval(str(valor))
    except:
        return [str(valor)]

def similaridade_texto(a, b):
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
    if not lista_candidato or not texto_vaga:
        return 0.0
    lista = safe_list(lista_candidato)
    texto = str(texto_vaga).lower()
    if not lista:
        return 0.0
    matches = sum(1 for item in lista if str(item).lower() in texto)
    return round(matches / len(lista), 4)

def nivel_para_numero(nivel):
    mapa = {
        "entrando na área": 0, "estagio": 0, "estágio": 0,
        "junior": 1, "júnior": 1,
        "pleno": 2,
        "senior": 3, "sênior": 3,
    }
    return mapa.get(str(nivel).lower().strip(), 1)

# ── 4. Formato da requisição ──
class MatchRequest(BaseModel):
    candidato_id: str
    vaga_id: str

# ── 5. Rotas da API ──

@app.get("/teste")
def testar_conexao():
    return {"status": "Sucesso!", "mensagem": "O Android conversou com o modelo no PC!"}

@app.post("/score")
def calcular_score(request: MatchRequest):
    try:
        c_id = request.candidato_id
        v_id = request.vaga_id

        # ── BUSCA NO FIRESTORE ──
        
        # 1. Busca Candidato
        cand_ref = db.collection("Candidatos").document(c_id).get()
        if cand_ref.exists:
            cand = cand_ref.to_dict()
        else:
            cand_docs = db.collection("Candidatos").where("id", "==", c_id).stream()
            cand = next((d.to_dict() for d in cand_docs), None)
            
        if not cand:
            raise HTTPException(status_code=444, detail="Candidato não encontrado")

        # 2. Busca Questionário
        quest_ref = db.collection("Questionario").document(c_id).get()
        if quest_ref.exists:
            quest = quest_ref.to_dict()
        else:
            quest_docs = db.collection("Questionario").where("uid", "==", c_id).stream()
            quest = next((d.to_dict() for d in quest_docs), None)
            
        if not quest:
            raise HTTPException(status_code=444, detail="Questionário não encontrado")

        # 3. Busca Vaga
        vaga_ref = db.collection("Vagas").document(v_id).get()
        if vaga_ref.exists:
            vaga = vaga_ref.to_dict()
        else:
            vaga_docs = db.collection("Vagas").where("id", "==", v_id).stream()
            vaga = next((d.to_dict() for d in vaga_docs), None)
            
        if not vaga:
            raise HTTPException(status_code=444, detail="Vaga não encontrada")


        # ── EXTRAÇÃO E CÁLCULO DAS FEATURES ──
        f_area = similaridade_texto(quest.get("areaDesejada", ""), vaga.get("area", vaga.get("titulo", "")))
        f_stack = match_lista(quest.get("stackInteresse", ""), vaga.get("stack", ""))
        f_setor = match_lista(quest.get("setorInteresse", ""), vaga.get("setor", ""))
        
        nivel_cand = nivel_para_numero(quest.get("nivelAtual", "junior"))
        nivel_vaga = nivel_para_numero(vaga.get("nivelExperiencia", "Pleno"))
        f_nivel = max(0.0, 1.0 - abs(nivel_cand - nivel_vaga) / 3)
        
        f_modalidade = 1.0 if str(quest.get("modalidade", "")).lower().strip() == str(vaga.get("modalidade", "")).lower().strip() else 0.0
        f_cidade = similaridade_texto(cand.get("cidade", ""), vaga.get("cidade", ""))
        
        try:
            # CORREÇÃO: Chave com acento conforme FirebaseConstants.java
            sal_cand_val = cand.get("pretensãoSalarial", 0)
            sal_cand = float(sal_cand_val if sal_cand_val else 0)
            
            sal_vaga_val = vaga.get("salario", 0)
            sal_vaga = float(sal_vaga_val if sal_vaga_val else 0)
            
            f_salario = min(1.0, sal_vaga / sal_cand) if sal_vaga > 0 and sal_cand > 0 else 0.5
        except Exception as e:
            print(f"Erro calculo salario: {e}")
            f_salario = 0.5

        # ── PREDIÇÃO ──
        features = [f_area, f_stack, f_setor, f_nivel, f_modalidade, f_cidade, f_salario]
        dados_entrada = np.array([features])
        
        score_bruto = modelo.predict(dados_entrada)[0]
        porcentagem_match = 1 / (1 + math.exp(-score_bruto))
        match_score_final = round(porcentagem_match * 100)

        # Log para você acompanhar no terminal
        print(f"DEBUG: Cand {c_id[:5]} -> Vaga {v_id[:5]} | Score: {match_score_final}% | Features: {features}")

        return {
            "status": "Sucesso",
            "vaga_id": v_id,
            "candidato_id": c_id,
            "score": match_score_final
        }
        
    except HTTPException as http_err:
        print(f"Erro HTTP: {http_err.detail}")
        raise http_err
    except Exception as e:
        print(f"Erro Interno: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)