# mteb_memoreba

## Executar com Docker Compose

Para executar o script com Docker Compose, use
```bash
sudo docker compose up -d
```

Para deletar o container e a imagem com Docker Compose, use
```bash
sudo docker compose down -rmi 'all'
```

Para subir tudo e logo depois derrubar e deletar tudo, use
```bash
sudo docker compose up && sudo docker compose down --rmi 'all'
```
Lembre-se que com esse último comando o container e a imagem não serão deletados se a execução do script for interrompida ou finalizada sem sucesso.

Os resultados do script estarão na pasta code/results.

## Executar com venv

Crianção da venv
```bash
python3 -m venv .venv
``` 

Ativação da venv
```bash
source .venv/bin/activate
``` 

Instalação dos requisitos
```bash
pip install -r requirements.txt
``` 
Execução do script
```bash
Python3 code/ScriptMTEB.py
``` 

Desativação da venv
```bash
deactivate
``` 

Quando necessário: atualização dos requisitos
```bash
pip freeze > requirements.txt
``` 