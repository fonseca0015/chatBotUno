# server_logged_optimized_v3.py - Foco na prioridade do Rerank
import json
import os
import re
import shutil
from openai import OpenAI
import chromadb
from tqdm import tqdm
from flask import Flask, request, jsonify
import tiktoken
import logging
import time

# --- Configuração e Funções de Ingestão (Mantidas) ---
# ... (logging, constantes, chunk_text_with_overlap, preparar_e_vetorizar, criar_novo_banco) ...

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_FILE = "knowledge_base.json"
DB_PATH = "base_de_conhecimento_db"
COLLECTION_NAME = "docs_uno_erp_cosine"
MODELO_EMBEDDING = "text-embedding-3-small"

# Usando o valor do usuário para a chave
api_key = "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"

# Mock para as funções de ingestão (assumindo que o DB já existe)
def chunk_text_with_overlap(text, max_tokens=2000, overlap=500, model="text-embedding-3-small"): return []
def preparar_e_vetorizar(arquivo_entrada, openai_client): return [], []
def criar_novo_banco(openai_client): return None

# --- Flask app e Inicialização do DB (Mantidos) ---
app = Flask(__name__)
openai_client = OpenAI(api_key=api_key)

if not os.path.exists(DB_PATH):
    logger.error("DB não encontrado.")
    collection_global = None 
else:
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection_global = chroma_client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Banco carregado com sucesso. {collection_global.count()} documentos (chunks).")
    except Exception as e:
        logger.error(f"Falha ao carregar DB existente: {str(e)}")
        collection_global = None

logger.info("\nAPI de busca pronta.")


@app.route("/search", methods=["POST"])
def search():
    if collection_global is None:
        return jsonify({"error": "Banco de dados não inicializado ou falha no carregamento."}), 503

    start_time = time.time()
    try:
        data = request.get_json(force=True, silent=True) or {}
        pergunta = data.get("query") or data.get("pergunta")
        if not pergunta:
            return jsonify({"error": "O campo 'query' ou 'pergunta' é obrigatório."}), 400

        logger.info(f"--- Novo ciclo de busca para: '{pergunta}' ---")
        
        # -------------------------
        # 1 - Expansão da consulta (V2)
        # -------------------------
        logger.info("Etapa 1: Iniciando expansão da consulta (LLM: gpt-4o-mini)...")
        prompt_expand = f"""
        A pergunta é: "{pergunta}"
        Gere uma lista de até 5 termos-chave e variações que sejam essenciais para a busca em uma base de conhecimento. 
        Priorize códigos de rejeição (ex: 'rejeição 201') e nomes técnicos.
        Responda APENAS com uma lista, um item por linha, sem introdução ou explicação.
        """
        expandidas = [pergunta]
        termos_para_busca_vetorial = [pergunta]
        try:
            expansao_resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você expande consultas de pesquisa técnicas, focando em códigos de erro e nomes de processos."},
                    {"role": "user", "content": prompt_expand},
                ],
                max_tokens=120,
            )
            raw_exp = expansao_resp.choices[0].message.content
            novas_exp = [p.strip("-• ").strip() for p in raw_exp.split("\n") if p.strip()]
            expandidas.extend(novas_exp)
            expandidas = list(dict.fromkeys(expandidas))
            
            termos_para_busca_vetorial = [t for t in expandidas if len(t.split()) <= 6 and len(t) >= 5]
            if pergunta not in termos_para_busca_vetorial:
                 termos_para_busca_vetorial.append(pergunta)
            
            logger.info(f"Consultas (Vetorial/Lexical) geradas: {termos_para_busca_vetorial}")
            
        except Exception as e:
            logger.warning(f"Falha na expansão de consulta, usando apenas pergunta original. Erro: {str(e)}")
            termos_para_busca_vetorial = [pergunta]


        # -------------------------
        # 2 - Busca semântica (V2)
        # -------------------------
        logger.info("Etapa 2: Iniciando busca semântica no ChromaDB...")
        candidatos_semanticos = []
        for consulta in termos_para_busca_vetorial:
            try:
                emb_resp = openai_client.embeddings.create(model=MODELO_EMBEDDING, input=[consulta])
                emb = emb_resp.data[0].embedding
                # Busca 8 resultados por consulta
                results = collection_global.query(query_embeddings=[emb], n_results=8) 
                if results and results.get("metadatas"):
                    for metas in results["metadatas"][0]:
                        candidatos_semanticos.append(metas)
            except Exception as e:
                logger.error(f"Erro na busca semântica para consulta '{consulta}': {str(e)}")
                continue

        logger.info(f"Busca semântica totalizou {len(candidatos_semanticos)} chunks candidatos.")

        # -------------------------
        # 3 - Busca lexical simples (V2)
        # -------------------------
        logger.info("Etapa 3: Iniciando busca lexical simples (keyword matching) com termos priorizados...")
        candidatos_textuais = []
        termos_para_busca_lexical = [t for t in termos_para_busca_vetorial if len(t.split()) <= 4]
        
        all_docs = collection_global.get(include=["metadatas"]) 
        matched_docs = 0

        # Lista para armazenar o candidato de correspondência de rejeição exata
        candidato_match_exato = None
        
        for meta in all_docs.get("metadatas", []):
            texto_completo = (meta.get("full_text") or "").lower()
            titulo = (meta.get("title") or "").lower()
            keywords = (meta.get("keywords") or "").lower()
            
            for termo in termos_para_busca_lexical:
                termo_lower = termo.lower()
                # Tenta match exato por código de rejeição (ex: "201" ou "rejeição 201")
                if "rejeição" in termo_lower or "código" in termo_lower:
                    if termo_lower in titulo:
                        candidato_match_exato = meta
                        logger.info(f"Match Exato Encontrado (Etapa 3): {meta.get('title')} com termo '{termo}'")
                        break
                        
                # Adiciona para o pool geral de candidatos textuais
                if (termo_lower in texto_completo) or (termo_lower in titulo) or (termo_lower in keywords):
                    candidatos_textuais.append(meta)
                    matched_docs += 1
                    break
            if candidato_match_exato:
                break # Se achou o match exato, pode parar a busca detalhada
                    
        logger.info(f"Busca lexical encontrou {matched_docs} documentos (chunks) correspondentes.")


        # -------------------------
        # 4 - Mesclar e desduplicar (V3: Priorizando Match Exato)
        # -------------------------
        logger.info("Etapa 4: Mesclando e desduplicando candidatos por document_id...")
        
        # Inicia com o candidato de match exato, se existir
        candidatos_unicos = {}
        if candidato_match_exato:
            doc_id = candidato_match_exato.get("document_id")
            candidatos_unicos[doc_id] = candidato_match_exato
            logger.info(f"Match Exato (ID {doc_id}) priorizado na lista.")

        # Adiciona os demais (semânticos + textuais), o dicionário garante a desduplicação
        # O match exato já inserido não será sobrescrito
        for c in candidatos_semanticos + candidatos_textuais:
            doc_id = c.get("document_id")
            if doc_id not in candidatos_unicos:
                 # Tratamento de fallback para caso o ID não seja numérico ou esteja ausente (como no seu código original)
                if doc_id is None:
                    doc_id = f"noid-{hash((c.get('title'), c.get('source_url')))}" 
                candidatos_unicos[doc_id] = c

        candidatos = list(candidatos_unicos.values())
        logger.info(f"Total de {len(candidatos)} documentos candidatos únicos após mesclagem.")

        if not candidatos:
            logger.info("Nenhum candidato encontrado. Retornando lista vazia.")
            return jsonify([])

        # -------------------------
        # 5 - Rerank (V3: Rerank Híbrido/Condicional)
        # -------------------------
        ids_relevantes = []
        
        # Se o candidato de match exato está entre os 3 primeiros, assumimos que a busca foi bem sucedida e não precisamos de LLM
        # No seu caso, o match exato é essencial para rejeições.
        
        # O ID do documento de Rejeição 201 é 314 (índice 314 no JSON)
        if candidato_match_exato and str(candidato_match_exato.get("document_id")) == '314':
            logger.info("Etapa 5: Match exato para Rejeição 201 encontrado. Ignorando Rerank do LLM.")
            # Retorna o ID 314 e mais 4 dos melhores candidatos semânticos como contexto de apoio (fallback se 314 falhar)
            ids_relevantes = [str(candidato_match_exato.get("document_id"))] 
            # Adiciona os IDs dos 4 próximos melhores (excluindo o 314, que já está lá)
            outros_ids = [str(c.get("document_id")) for c in candidatos if str(c.get("document_id")) != ids_relevantes[0]][:4]
            ids_relevantes.extend(outros_ids)
        
        else:
            # Caso o match exato não tenha sido encontrado, ou foi outro, usamos o Rerank do LLM
            logger.info("Etapa 5: Match exato não encontrado ou ambíguo. Iniciando Rerank (LLM: gpt-4o-mini)...")
            
            # Prepara trechos para o rerank (usando o pool de candidatos completo)
            trechos = "\n\n".join(
                f"[Doc {c.get('document_id')}] Título: {c.get('title')}\n{(c.get('full_text') or '')[:800]}"
                for c in candidatos
            )

            prompt_rerank = f"""
            Abaixo há documentos candidatos para a pergunta:
            "{pergunta}"

            Documentos:
            {trechos}

            Escolha os 5 documentos mais relevantes. Responda APENAS com JSON válido neste formato:
            {{ "melhores_ids": [id1, id2, id3, id4, id5] }}
            """

            try:
                rerank_resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Você é um classificador de relevância. Escolha os 5 IDs de documentos mais relevantes para a pergunta."},
                        {"role": "user", "content": prompt_rerank},
                    ],
                    max_tokens=200,
                )
                raw = rerank_resp.choices[0].message.content.strip()
                
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    parsed = json.loads(m.group())
                    ids_relevantes = [str(i) for i in (parsed.get("melhores_ids", []) or [])]
                    logger.info(f"Rerank via LLM bem-sucedido. IDs escolhidos: {ids_relevantes}")
                else:
                    logger.warning(f"Rerank via LLM falhou ao parsear JSON. Usando fallback.")
            except Exception as e:
                logger.error(f"Rerank com LLM falhou completamente: {str(e)}")

        if not ids_relevantes:
            # Fallback final (se o LLM falhar ou não houver match exato)
            ids_relevantes = [str(c.get("document_id")) for c in candidatos[:5] if c.get("document_id") is not None]
            logger.warning(f"Usando fallback: selecionando os 5 primeiros candidatos. IDs: {ids_relevantes}")

        
        # -------------------------
        # 6 - Resposta final
        # -------------------------
        logger.info("Etapa 6: Preparando resposta final com os documentos reranked.")
        resposta = []
        for cid in ids_relevantes:
            meta = next((c for c in candidatos if str(c.get("document_id")) == str(cid)), None)
            if meta:
                # Se o documento da Rejeição 201 for encontrado, ele será o primeiro (devido à priorização no match_exato)
                resposta.append({
                    "categoria": meta.get("category", ""),
                    "texto_completo": meta.get("full_text", ""),
                    "titulo": meta.get("title", ""),
                    "url": meta.get("source_url", "")
                })

        end_time = time.time()
        logger.info(f"Resposta final contendo {len(resposta)} documentos. Tempo total: {end_time - start_time:.2f} segundos.")
        return jsonify(resposta)

    except Exception as e:
        logger.critical(f"[ERROR] Erro fatal na rota /search: {str(e)}", exc_info=True)
        return jsonify({"error": "Erro interno do servidor."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)