import pandas as pd
import json
import os

# folder with json
pasta = r"C:\Users\natyj\Documents\ICO\Program\ico-fil-rouge\BaseDeDonnees\PetitBase"

# all json
for arquivo in os.listdir(pasta):
    
    if arquivo.endswith(".json"):
        
        caminho_json = os.path.join(pasta, arquivo)
        
        # open json
        with open(caminho_json, "r", encoding="utf-8") as f:
            dados = json.load(f)

        # transform in DataFrame
        df = pd.DataFrame(dados)

        # excel name
        nome_excel = arquivo.replace(".json", ".xlsx")
        caminho_excel = os.path.join(pasta, nome_excel)

        # salve
        df.to_excel(caminho_excel, index=False)

        print(f"Arquivo criado: {nome_excel}")