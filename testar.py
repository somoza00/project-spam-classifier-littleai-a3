from spam_classifier import predict_auto, load_model

# Pré-carrega os dois modelos na memória
load_model("en")
load_model("pt")

print("=== Testador de Spam (EN + PT-BR) ===")
print("Digite uma mensagem e pressione Enter.")
print("O idioma é detectado automaticamente.")
print("Digite 'sair' para encerrar.\n")

while True:
    texto = input("Mensagem: ").strip()
    if texto.lower() == "sair":
        break
    if not texto:
        continue

    resultado = predict_auto([texto])[0]
    lang_nome = "Português" if resultado["lang"] == "pt" else "English"

    if resultado["prediction"] == "spam":
        print(f"  → SPAM [{lang_nome}]\n")
    else:
        print(f"  → HAM  [{lang_nome}] (não é spam)\n")
