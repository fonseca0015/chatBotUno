import json
import re

# Caminhos de entrada e saída
input_path = "base_limpa.json"
output_path = "base_limpa_sem_espacos.json"

def limpar_texto(texto: str) -> str:
    # Remove tags HTML (tudo entre <...>)
    texto = re.sub(r"<[^>]*>", "", texto)

    # Remove espaços no início e final
    texto = texto.strip()

    # Substitui múltiplos espaços por apenas um
    texto = re.sub(r"\s{2,}", " ", texto)

    # Remove espaços no final das linhas (se houver quebras)
    linhas = [linha.rstrip() for linha in texto.splitlines()]
    texto = "\n".join(linhas)

    return texto

# Lê o JSON original
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Aplica a limpeza no campo "Conteudo"
if isinstance(data, list):
    for item in data:
        if "Conteudo" in item and isinstance(item["Conteudo"], str):
            item["Conteudo"] = limpar_texto(item["Conteudo"])

# Salva o JSON limpo
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Arquivo limpo salvo como:", output_path)
