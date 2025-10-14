import json
import math

# Caminho do arquivo JSON original
with open("base_limpa.json", "r", encoding="utf-8") as f:
    dados = json.load(f)

# Filtra apenas os registros cujo conteúdo é válido (não vazio, não NaN)
dados_filtrados = [
    item for item in dados
    if item.get("Conteudo") not in [None, "", "NaN"]
    and not (isinstance(item.get("Conteudo"), float) and math.isnan(item["Conteudo"]))
]

# Salva em um novo arquivo JSON limpo
with open("base_limpa_sem_nan.json", "w", encoding="utf-8") as f:
    json.dump(dados_filtrados, f, ensure_ascii=False, indent=2)

print(f"Total antes: {len(dados)} | Total depois da limpeza: {len(dados_filtrados)}")
