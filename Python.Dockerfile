# Usar versão mais nova do Python
FROM python:3.10-bullseye

# Diretório de trabalho do container
WORKDIR /mteb-amanda-container

# Copiar os requirements para o container
COPY ./requirements.txt ./

# Instala as dependências usando Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código para o container
COPY ./ ./

# Muda o diretório de trabalho
WORKDIR /mteb-amanda-container/code-container

# Setup the command to run when the container starts
CMD ["python3", "ScriptMTEB.py"]