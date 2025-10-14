import pandas as pd
import json

# Carrega o CSV limpo
df = pd.read_csv("extracao_limpa_sem_emoji.csv", encoding="utf-8")

# Converte cada linha para um dicionário JSON
dados_json = df.to_dict(orient="records")

# Salva em arquivo JSON
with open("base_limpa.json", "w", encoding="utf-8") as f:
    json.dump(dados_json, f, ensure_ascii=False, indent=2)