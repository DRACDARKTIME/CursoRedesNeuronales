# Entorno de entrenamiento (`house-tf`)

Entorno **local** para entrenar la red. La app desplegada NO usa este entorno: usa
`../mi-proyecto-azure/requirements.txt` dentro de su Dockerfile.

| Componente | Version |
|---|---|
| Python | 3.11 |
| TensorFlow (GPU) | 2.16.1 (`tensorflow[and-cuda]`, CUDA 12.3) |
| Keras | 3.3.3 |
| MLflow | 2.17.2 |
| kagglehub | 1.x |
| DagsHub | 0.7.x |
| JupyterLab | 4.x |

## Uso diario

```zsh
# si `conda` no existe en tu shell, habilitalo una sola vez:
~/miniconda3/bin/conda init zsh && exec zsh

conda activate house-tf
jupyter lab          # el kernel se llama "Python (house-tf)"
```

Para borrar y recrear el entorno desde cero:

```zsh
conda env remove -n house-tf
conda env create -f environment.yml
```

## GPU (RTX 4060)

TensorFlow no encuentra las librerías CUDA que instala `pip` si no están en
`LD_LIBRARY_PATH`. Este entorno ya trae el hook
`etc/conda/activate.d/tensorflow_cuda.sh`, que las añade al activarlo.
Si recreas el entorno en otra máquina, vuelve a crear el hook:

```zsh
ENV=~/miniconda3/envs/house-tf
mkdir -p $ENV/etc/conda/activate.d $ENV/etc/conda/deactivate.d
cat > $ENV/etc/conda/activate.d/tensorflow_cuda.sh <<'EOF'
for _d in "$CONDA_PREFIX"/lib/python3.11/site-packages/nvidia/*/lib; do
  [ -d "$_d" ] || continue
  case ":$LD_LIBRARY_PATH:" in
    *":$_d:"*) ;;
    *) LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
done
export LD_LIBRARY_PATH
unset _d
EOF
```

Comprobar:

```zsh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Si `pip` se queda colgado

Tu `pip` global tiene `extra-index-url = https://pypi.ngc.nvidia.com`, que bloquea
las instalaciones. Ignóralo con:

```zsh
PIP_CONFIG_FILE=/dev/null pip install <paquete>
```

## Credenciales de tracking (no subir a git)

Crea `EntrenamientoRed/.env`:

```dotenv
MLFLOW_TRACKING_URI=https://dagshub.com/<usuario>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<usuario>
MLFLOW_TRACKING_PASSWORD=<token_de_dagshub>
```

Y en el notebook:

```python
from dotenv import load_dotenv
import mlflow, dagshub

load_dotenv()
dagshub.init(repo_owner="<usuario>", repo_name="<repo>", mlflow=True)
mlflow.set_experiment("compra-casa")
```

## Consistencia con la app

- Guarda el modelo como `model/best_model.keras` (formato nativo Keras).
- Si usas `StandardScaler`, **no** lo picklees con otra versión de scikit-learn que
  la de la app (aquí hay 1.9.x, la app usa 1.4.2). Alternativas seguras:
  normalizar con una capa `keras.layers.Normalization` dentro del modelo, o guardar
  `mean_` y `scale_` en un `.npz`/`.json` y aplicarlos a mano en `app.py`.
