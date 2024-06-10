from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from diffusers import DiffusionPipeline
from PIL import Image
import torch
import os
import io
import base64
import time
from cachetools import LRUCache, cached

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo utilizado: {device}")

# Rutas locales
ruta_carpeta_principal = 'C:/Users/Usuario01/PythonProjects/13_fastapi_modelo/input_folder'
model_path = 'C:/Users/Usuario01/PythonProjects/13_fastapi_modelo/modelo'

# Verificar que las rutas existen
if not os.path.exists(ruta_carpeta_principal):
    print(f"La ruta {ruta_carpeta_principal} no existe.")
if not os.path.exists(model_path):
    print(f"La ruta {model_path} no existe.")

# Cargando el modelo desde la ruta local
try:
    print("Cargando el modelo...")
    arte = DiffusionPipeline.from_pretrained(model_path).to(device)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")

# Identificando las subcarpetas de estilos
try:
    subcarpetas = [f for f in os.listdir(ruta_carpeta_principal) if os.path.isdir(
        os.path.join(ruta_carpeta_principal, f))]
    print(f"Subcarpetas encontradas: {subcarpetas}")
except Exception as e:
    print(f"Error al identificar subcarpetas: {str(e)}")

# Configuración del caché
cache = LRUCache(maxsize=100)


@cached(cache)
def generar_imagen_cache(style):
    print(f"Generando imagen para el estilo: {style}")
    subcarpeta = os.path.join(ruta_carpeta_principal, style)
    archivos_en_subcarpeta = os.listdir(subcarpeta)
    archivos_de_imagen = [archivo for archivo in archivos_en_subcarpeta if archivo.endswith(
        ('.jpg', '.jpeg', '.png', '.gif'))]

    if not archivos_de_imagen:
        raise ValueError("No se encontraron imágenes en la subcarpeta.")

    ruta_imagen = os.path.join(subcarpeta, archivos_de_imagen[0])
    imagen = Image.open(ruta_imagen).convert("RGB")

    # Reduciendo la resolución para acelerar el procesamiento y disminuir el uso de memoria
    target_size = (64, 64)  # Reduciendo aún más la resolución
    imagen = imagen.resize(target_size)

    # Generando una nueva imagen usando el modelo de difusión
    imagen_procesada = arte(imagen).images[0]

    # Guardando la imagen en un objeto BytesIO
    img_io = io.BytesIO()
    imagen_procesada.save(img_io, 'PNG')
    img_io.seek(0)

    # Convirtiendo la imagen a base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return img_base64


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "estilos": subcarpetas})


@app.post("/generate", response_class=HTMLResponse)
async def generate_image(request: Request, style: str = Form(...)):
    start_time = time.time()

    try:
        print(f"Recibida solicitud para generar imagen en estilo: {style}")
        img_base64 = generar_imagen_cache(style)
        elapsed_time = time.time() - start_time
        print(f"Tiempo de procesamiento: {elapsed_time} segundos")

        return templates.TemplateResponse("index.html", {"request": request, "estilos": subcarpetas, "imagen_generada": img_base64, "tiempo_procesamiento": elapsed_time})
    except Exception as e:
        print(f"Error al procesar la imagen: {str(e)}")
        return HTMLResponse(content=f"Error al procesar la imagen: {str(e)}", status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
