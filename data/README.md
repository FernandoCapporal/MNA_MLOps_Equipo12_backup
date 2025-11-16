# 🧊 Gestión de Datos en S3 con DVC (Data Version Control) usando boto3 + dotenv

Este documento explica el flujo completo para almacenar, versionar y sincronizar datos en **Amazon S3** utilizando **DVC** y **boto3**, con las credenciales gestionadas a través de **dotenv** (.env).  
El objetivo es mantener control de versiones sobre grandes datasets o modelos sin saturar el repositorio Git y sin depender de `aws configure`.

---

## 🚀 1. Prerrequisitos

Antes de comenzar, asegúrate de tener lo siguiente instalado y configurado:

### 🧰 Dependencias locales

| Herramienta | Instalación |
|--------------|-------------|
| **Git** | `sudo apt install git` o `brew install git` |
| **DVC (con soporte S3)** | `pip install "dvc[s3]"` |
| **boto3** | `pip install boto3` |
| **python-dotenv** | `pip install python-dotenv` |
| **Repositorio Git** | Inicializado con `git init` |

---

## 🔐 2. Configurar credenciales AWS con dotenv

En lugar de usar `aws configure`, definiremos las credenciales de AWS en un archivo `.env`.

Crea un archivo llamado `.env` en la raíz del proyecto con el siguiente contenido:

```env
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=nombre-de-tu-bucket
```

> ⚠️ **Importante:**  
> Nunca subas este archivo a Git.  
> Añádelo a `.gitignore` con:
> ```bash
> echo ".env" >> .gitignore
> ```

---

## 🌎 3. Cargar variables de entorno con `python-dotenv`

Puedes usar las credenciales en cualquier script Python con:

```python
from dotenv import load_dotenv
import os
import boto3

# Cargar las variables desde el archivo .env
load_dotenv()

# Crear cliente S3 usando boto3
s3 = boto3.client('s3')

# Verificar conexión
buckets = s3.list_buckets()
print("Buckets disponibles:", [b['Name'] for b in buckets['Buckets']])
```

Si ves los nombres de tus buckets, la conexión es exitosa ✅

---

## 🪣 4. Crear o usar un bucket S3

Si necesitas crear un bucket directamente desde Python:

```python
s3.create_bucket(Bucket=os.getenv("S3_BUCKET_NAME"), CreateBucketConfiguration={
    'LocationConstraint': os.getenv("AWS_DEFAULT_REGION")
})
```

O desde terminal:
```bash
aws s3 mb s3://nombre-de-tu-bucket --region us-east-1
```

Verifica:
```bash
aws s3 ls s3://nombre-de-tu-bucket
```

---

## 🧬 5. Inicializar DVC en el proyecto

Desde la raíz de tu repositorio:

```bash
dvc init
```

Esto crea el directorio `.dvc/` y los archivos `.dvc/config`.

Agrega DVC al control de versiones:

```bash
git add .dvc .gitignore
git commit -m "Initialize DVC"
```

---

## 🌐 6. Conectar DVC con tu bucket S3

Configura el *remote storage* de DVC apuntando a tu bucket (usando la variable del `.env`):

```bash
dvc remote add -d s3remote s3://$(grep S3_BUCKET_NAME .env | cut -d '=' -f2)/data
```

Confirma que quedó registrado:

```bash
dvc remote list
```

---

## 🗃️ 7. Versionar un archivo o dataset

Ejemplo: quieres versionar `data/raw/insurance_company_original.csv`

```bash
dvc add data/raw/insurance_company_original.csv
```
Seguramente primero hay que borrar el regitro que git ya había apuntado producto del git add .dvc:
```bash
git rm -r --cached 'data/raw/insurance_company_original.csv'
```

Posteriormente volver a tirar dvc add ... Esto genera un archivo `.dvc` con la metadata de versionado:
`data/raw/insurance_company_original.csv.dvc`

Agrégalo al repositorio:

```bash
git add data/raw/insurance_company_original.csv.dvc .gitignore
git commit -m "Track insurance dataset with DVC"
```

---

## ☁️ 8. Subir los datos a S3

Antes de ejecutar `dvc push`, exporta las variables del `.env`:

```bash
export $(grep -v '^#' .env | xargs)
```

Luego sube los datos:

```bash
dvc push
```

Esto transfiere los archivos físicos al bucket S3 definido.

Si ves un error como:

```
The bucket you are attempting to access must be addressed using the specified endpoint
```

Verifica la región con:
```bash
aws s3api get-bucket-location --bucket nombre-de-tu-bucket
```

Y actualiza el endpoint en DVC si es necesario:
```bash
dvc remote modify s3remote endpointurl https://s3.us-east-1.amazonaws.com
```

---

## 🔄 9. Recuperar los datos (pull)

En otra máquina o entorno limpio:

```bash
git clone git@github.com:FernandoCapporal/MNA_MLOps_Equipo12_backup.git
cd MNA_MLOps_Equipo12_backup
export $(grep -v '^#' .env | xargs)
dvc pull
```

Esto descargará los datos versionados desde S3 utilizando las credenciales cargadas con dotenv.

---

## 📚 10. Comandos útiles

| Comando | Descripción |
|----------|-------------|
| `dvc status` | Muestra qué archivos están desactualizados respecto al remoto |
| `dvc push` | Sube los datos al almacenamiento remoto |
| `dvc pull` | Descarga los datos versionados desde S3 |
| `dvc gc` | Limpia versiones antiguas no referenciadas |
| `dvc repro` | Ejecuta pipelines definidos con dependencias (`.dvc.yaml`) |

---

## 🧠 11. Buenas prácticas

- Nunca subas directamente los archivos grandes al repositorio Git.  
  Usa `dvc add` y deja que DVC los maneje.
- Mantén las credenciales en `.env` y asegúrate de que esté en `.gitignore`.
- Puedes combinar `boto3` y `DVC` para automatizar la creación de buckets o la validación de subidas.
- Usa etiquetas (`git tag`) para marcar versiones importantes de datasets y modelos.

---

## 🧩 12. Ejemplo de flujo completo

```bash
# Inicializar proyecto
git init
dvc init

# Agregar datos
mkdir -p data/raw
cp dataset.csv data/raw/
dvc add data/raw/dataset.csv

# Configurar S3 (usando variable del .env)
export $(grep -v '^#' .env | xargs)
dvc remote add -d s3remote s3://$S3_BUCKET_NAME/data
dvc push

# Versionar cambios
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "First version of dataset"
```

---

## 🧾 13. Referencias

- [DVC Docs: Remote Storage (S3)](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)  
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)  
- [python-dotenv Documentation](https://pypi.org/project/python-dotenv/)  
- [AWS CLI Reference](https://docs.aws.amazon.com/cli/latest/reference/)
