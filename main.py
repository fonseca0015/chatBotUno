import json
import os
import re
import shutil
from openai import OpenAI
import chromadb
from tqdm import tqdm
from flask import Flask, request, jsonify

# --- Configurações Globais ---
RAW_DATA_FILE = 'base_limpa_ajustada.json'
DB_PATH = "base_de_conhecimento_db"
COLLECTION_NAME = "docs_uno_erp_cosine"
MODELO_EMBEDDING = "text-embedding-3-small"
api_key = os.environ.get("OPENAI_API_KEY")

# --- Funções de Preparação e Ingestão ---
# Estas funções só serão chamadas se o banco de dados não existir.
def preparar_e_vetorizar(arquivo_entrada, openai_client):
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_chunks_to_embed = []
    all_metadatas = []

    print("Iniciando o processo de chunking e preparação de dados...")
    for doc_id, doc in enumerate(data):
        titulo = doc.get('Titulo', '')
        conteudo = doc.get('Conteudo', '')
        palavras_chave = doc.get('Palavras Chave', '')
        
        chunks = conteudo.split('\n\n')
        for chunk_id, chunk_text in enumerate(chunks):
            cleaned_chunk = chunk_text.strip()
            if len(cleaned_chunk) >= 50:
                text_to_embed = f"Título: {titulo} | Palavras-chave: {palavras_chave} | Conteúdo: {cleaned_chunk}"
                all_chunks_to_embed.append(text_to_embed)
                
                metadata = {
                    'original_text': cleaned_chunk, 'source_url': doc.get('URL', ''),
                    'category': doc.get('Categoria', ''), 'title': titulo,
                    'keywords': palavras_chave, 'document_id': doc_id, 'chunk_id': chunk_id
                }
                all_metadatas.append(metadata)

    print(f"Chunking concluído. {len(all_chunks_to_embed)} chunks criados.")
    print("Iniciando a vetorização com a OpenAI...")

    todos_embeddings = []
    batch_size = 100

    for i in tqdm(range(0, len(all_chunks_to_embed), batch_size), desc="Vetorizando Chunks"):
        batch = all_chunks_to_embed[i:i+batch_size]
        response = openai_client.embeddings.create(input=batch, model=MODELO_EMBEDDING)
        embeddings = [item.embedding for item in response.data]
        todos_embeddings.extend(embeddings)
        
    print("Vetorização concluída!")
    return todos_embeddings, all_metadatas

def criar_novo_banco(openai_client):
    """Executa todo o processo de criação do banco de dados do zero."""
    if os.path.exists(DB_PATH):
        print(f"Apagando banco de dados antigo em '{DB_PATH}'...")
        shutil.rmtree(DB_PATH)

    embeddings, metadatas = preparar_e_vetorizar(RAW_DATA_FILE, openai_client)

    print("Criando novo banco de dados ChromaDB com a métrica 'cosine'...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    
    ids = [str(i) for i in range(len(metadatas))]
    batch_size = 100
    for i in tqdm(range(0, len(ids), batch_size), desc="Populando o ChromaDB"):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"Banco de dados criado e populado com {collection.count()} documentos.")
    return collection

# --- Criação da API Flask ---
app = Flask(__name__)

if api_key == "SUA_CHAVE_API_VAI_AQUI" or not api_key:
    raise ValueError("A chave da API da OpenAI não foi configurada.")

openai_client = OpenAI(api_key=api_key)

# --- LÓGICA DE INICIALIZAÇÃO OTIMIZADA ---
if not os.path.exists(DB_PATH):
    print("Banco de dados não encontrado. Iniciando processo de criação...")
    collection_global = criar_novo_banco(openai_client)
else:
    print(f"Carregando banco de dados ChromaDB existente de '{DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection_global = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Banco carregado com sucesso. {collection_global.count()} documentos.")

print("\nAPI de busca pronta para iniciar.")

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        pergunta = data.get('query')
        if not pergunta: return jsonify({"error": "O campo 'query' é obrigatório."}), 400

        rejection_pattern = re.search(r'(rejeição|rej)\s*(\d+)', pergunta, re.IGNORECASE)
        filtro_ids = None
        
        if rejection_pattern:
            numero_rejeicao = rejection_pattern.group(2)
            keyword_to_find = f"rejeição {numero_rejeicao}"
            
            all_rejections = collection_global.get(where={"category": "Rejeições NFE"})
            matching_ids = []
            if all_rejections['ids']:
                for i, metadata in enumerate(all_rejections['metadatas']):
                    if keyword_to_find in metadata.get('keywords', ''):
                        matching_ids.append(all_rejections['ids'][i])
            
            if matching_ids:
                filtro_ids = matching_ids

        response = openai_client.embeddings.create(input=[pergunta], model=MODELO_EMBEDDING)
        query_vector = [response.data[0].embedding]

        if filtro_ids:
             results = collection_global.query(
                query_embeddings=query_vector, ids=filtro_ids, n_results=min(5, len(filtro_ids))
            )
        else:
            results = collection_global.query(
                query_embeddings=query_vector, n_results=3
            )
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

