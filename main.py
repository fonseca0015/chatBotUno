# server.py
import json
import os
import re
import shutil
from openai import OpenAI
import chromadb
from tqdm import tqdm
from flask import Flask, request, jsonify
import tiktoken

# --- Configurações Globais ---
RAW_DATA_FILE = "knowledge_base.json"
DB_PATH = "base_de_conhecimento_db"
COLLECTION_NAME = "docs_uno_erp_cosine"
MODELO_EMBEDDING = "text-embedding-3-small"

# --- SUA CHAVE DIRETA AQUI ---
api_key = "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"  # <<< COLE SUA CHAVE AQUI

# --- Helpers de chunking (mantive sua função) ---
def chunk_text_with_overlap(text, max_tokens=2000, overlap=500, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

# --- Ingestão e vetorização (mantive lógica original) ---
def preparar_e_vetorizar(arquivo_entrada, openai_client):
    with open(arquivo_entrada, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks_to_embed = []
    all_metadatas = []

    print("Iniciando chunking e preparação de dados...")
    for doc_id, doc in enumerate(data):
        titulo = doc.get("Titulo", "")
        conteudo = doc.get("Conteudo", "")
        palavras_chave = doc.get("Palavras Chave", "")

        chunks = chunk_text_with_overlap(conteudo, max_tokens=3000, overlap=500)
        for chunk_id, chunk_text in enumerate(chunks):
            cleaned_chunk = chunk_text.strip()
            if len(cleaned_chunk) >= 50:
                text_to_embed = f"Título: {titulo} | Palavras-chave: {palavras_chave} | Conteúdo: {cleaned_chunk}"
                all_chunks_to_embed.append(text_to_embed)
                metadata = {
                    "full_text": conteudo,
                    "chunk_text": cleaned_chunk,
                    "source_url": doc.get("URL", ""),
                    "category": doc.get("Categoria", ""),
                    "title": titulo,
                    "keywords": palavras_chave,
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                }
                all_metadatas.append(metadata)

    print(f"Chunking concluído. {len(all_chunks_to_embed)} chunks criados.")
    print("Iniciando vetorização com OpenAI...")

    todos_embeddings = []
    batch_size = 100
    for i in tqdm(range(0, len(all_chunks_to_embed), batch_size), desc="Vetorizando Chunks"):
        batch = all_chunks_to_embed[i:i + batch_size]
        response = openai_client.embeddings.create(input=batch, model=MODELO_EMBEDDING)
        embeddings = [item.embedding for item in response.data]
        todos_embeddings.extend(embeddings)

    print("Vetorização concluída!")
    return todos_embeddings, all_metadatas

def criar_novo_banco(openai_client):
    if os.path.exists(DB_PATH):
        print(f"Apagando banco de dados antigo em '{DB_PATH}'...")
        shutil.rmtree(DB_PATH)

    embeddings, metadatas = preparar_e_vetorizar(RAW_DATA_FILE, openai_client)

    print("Criando novo banco ChromaDB com métrica 'cosine'...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids = [str(i) for i in range(len(metadatas))]
    batch_size = 100
    for i in tqdm(range(0, len(ids), batch_size), desc="Populando o ChromaDB"):
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Banco criado e populado com {collection.count()} documentos.")
    return collection

# --- Flask app ---
app = Flask(__name__)
openai_client = OpenAI(api_key=api_key)

# --- Inicialização do DB (criacao ou carregamento) ---
if not os.path.exists(DB_PATH):
    print("DB não encontrado. Criando novo DB...")
    collection_global = criar_novo_banco(openai_client)
else:
    print(f"Carregando DB ChromaDB existente de '{DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection_global = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Banco carregado com sucesso. {collection_global.count()} documentos.")

print("\nAPI de busca pronta.")

# --- ROTA /search ---
@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json(force=True, silent=True) or {}
        # Aceita tanto 'query' quanto 'pergunta'
        pergunta = data.get("query") or data.get("pergunta")
        if not pergunta:
            return jsonify({"error": "O campo 'query' ou 'pergunta' é obrigatório."}), 400

        print(f"[SEARCH] pergunta recebida: {pergunta}")

        # -------------------------
        # 1 - Expansão da consulta
        # -------------------------
        prompt_expand = f"""
Gere até 5 variações curtas e termos relacionados que ajudem a buscar documentos técnicos internos
para a pergunta: "{pergunta}"
Responda com uma lista, um item por linha, sem explicar.
"""
        try:
            expansao_resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você expande consultas de pesquisa técnicas."},
                    {"role": "user", "content": prompt_expand},
                ],
                max_tokens=120,
            )
            raw_exp = expansao_resp.choices[0].message.content
            expandidas = [p.strip("-• ").strip() for p in raw_exp.split("\n") if p.strip()]
            expandidas = list(dict.fromkeys([pergunta] + expandidas))
        except Exception as e:
            print("[WARN] falha na expansão de consulta, usando apenas pergunta original:", str(e))
            expandidas = [pergunta]

        # -------------------------
        # 2 - Busca semântica
        # -------------------------
        candidatos_semanticos = []
        for consulta in expandidas:
            try:
                emb_resp = openai_client.embeddings.create(model=MODELO_EMBEDDING, input=[consulta])
                emb = emb_resp.data[0].embedding
                results = collection_global.query(query_embeddings=[emb], n_results=5)
                if results and results.get("metadatas"):
                    for metas in results["metadatas"][0]:
                        candidatos_semanticos.append(metas)
            except Exception as e:
                print("[WARN] erro na busca semântica para consulta:", consulta, str(e))
                continue

        # -------------------------
        # 3 - Busca lexical simples
        # -------------------------
        candidatos_textuais = []
        consulta_lower = pergunta.lower()
        try:
            all_docs = collection_global.get()
            for meta in all_docs.get("metadatas", []):
                texto = (meta.get("full_text") or "").lower()
                titulo = (meta.get("title") or "").lower()
                keywords = (meta.get("keywords") or "").lower()
                if (consulta_lower in texto) or (consulta_lower in titulo) or (consulta_lower in keywords):
                    candidatos_textuais.append(meta)
        except Exception as e:
            print("[WARN] erro ao recuperar todos os documentos para busca lexical:", str(e))

        # -------------------------
        # 4 - Mesclar e desduplicar
        # -------------------------
        candidatos_unicos = {}
        for c in candidatos_semanticos + candidatos_textuais:
            doc_id = c.get("document_id")
            if doc_id is None:
                doc_id = f"noid-{hash((c.get('title'), c.get('source_url')))}"
            candidatos_unicos[doc_id] = c

        candidatos = list(candidatos_unicos.values())

        if not candidatos:
            return jsonify([])

        # -------------------------
        # 5 - Rerank
        # -------------------------
        trechos = "\n\n".join(
            f"[Doc {c.get('document_id')}] Título: {c.get('title')}\n{(c.get('full_text') or '')[:800]}"
            for c in candidatos
        )

        prompt_rerank = f"""
Abaixo há documentos candidatos para a pergunta:
"{pergunta}"

Documentos:
{trechos}

Escolha os 3 documentos mais relevantes. Responda APENAS com JSON válido neste formato:
{{ "melhores_ids": [id1, id2, id3] }}
"""

        ids_relevantes = []
        try:
            rerank_resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um classificador de relevância."},
                    {"role": "user", "content": prompt_rerank},
                ],
                max_tokens=200,
            )
            raw = rerank_resp.choices[0].message.content.strip()
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                parsed = json.loads(m.group())
                ids_relevantes = parsed.get("melhores_ids", []) or []
        except Exception as e:
            print("[WARN] rerank falhou:", str(e))

        if not ids_relevantes:
            ids_relevantes = [c.get("document_id") for c in candidatos[:3]]

        # -------------------------
        # 6 - Resposta final
        # -------------------------
        resposta = []
        for cid in ids_relevantes:
            meta = candidatos_unicos.get(cid)
            if not meta:
                for c in candidatos:
                    if str(c.get("document_id")) == str(cid):
                        meta = c
                        break
            if meta:
                resposta.append({
                    "categoria": meta.get("category", ""),
                    "texto_completo": meta.get("full_text", ""),
                    "titulo": meta.get("title", ""),
                    "url": meta.get("source_url", "")
                })

        return jsonify(resposta)

    except Exception as e:
        print("[ERROR] erro na rota /search:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
