import re
import pandas as pd

df = pd.read_csv("extracao_limpa.csv", encoding="utf-8")

# Remove emojis da coluna Categoria
def remover_emojis(texto):
    if isinstance(texto, str):
        return re.sub(r"[^\w\s,.-]", "", texto).strip()
    return texto

df["Categoria"] = df["Categoria"].apply(remover_emojis)

df.to_csv("extracao_limpa_sem_emoji.csv", index=False, encoding="utf-8-sig")
