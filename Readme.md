# Documentation fastsdcpu-pip

## Table des matières

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Architecture du module](#architecture-du-module)
4. [Modes d'optimisation](#modes-doptimisation)
5. [Utilisation](#utilisation)
6. [Configuration avancée](#configuration-avancée)
7. [API Reference](#api-reference)
8. [Exemples pratiques](#exemples-pratiques)

---

## Introduction

`fastsdcpu-pip` est un module Python conçu pour exécuter des modèles **Stable Diffusion** de manière optimisée sur **CPU** et **AI PC** (PC avec accélérateurs IA). Il s'agit d'une adaptation du projet [fastsdcpu](https://github.com/rupeshs/fastsdcpu) sous forme de package pip.

### Caractéristiques principales

- **Optimisations CPU** : Utilisation de modèles LCM (Latent Consistency Models) pour une inférence rapide
- **Support OpenVINO** : Optimisations spécifiques pour les processeurs Intel et NPU
- **Support GGUF** : Modèles quantifiés pour une utilisation efficace de la mémoire
- **Modes multiples** : CLI, API Web, serveur MCP
- **Flexibilité** : Text-to-Image, Image-to-Image, ControlNet, LoRA, upscaling

### Prérequis

- Python ≥ 3.11
- CPU moderne (Intel/AMD/Apple Silicon)
- 8 GB RAM minimum (16 GB recommandé)
- Optionnel : NPU Intel pour accélération matérielle

---

## Installation

### Installation standard

```bash
pip install fastsdcpu-pip
```

### Installation depuis les sources

```bash
git clone https://github.com/Flyns157/fastsdcpu.git
cd fastsdcpu/pip
pip install -e .
```

### Dépendances principales

Le module installe automatiquement :
- `diffusers` : Pipelines Stable Diffusion
- `transformers` : Modèles de langage (CLIP, T5)
- `torch` : Framework de deep learning
- `openvino` + `optimum-intel` : Optimisations Intel
- `onnxruntime` : Runtime ONNX
- `peft` : Support LoRA
- `controlnet-aux` : Préprocesseurs ControlNet

---

## Architecture du module

### Structure des répertoires

```
fastsdcpu-pip/
├── configs/               # Fichiers de configuration
│   ├── lcm-models.txt        # Liste des modèles LCM
│   ├── lcm-lora-models.txt   # Liste des modèles LCM-LoRA
│   ├── openvino-lcm-models.txt
│   └── stable-diffusion-models.txt
├── fastsdcpu/             # Code source principal
│   ├── annotators/           # Préprocesseurs ControlNet
│   ├── api/                  # Serveurs API (Web, MCP)
│   ├── models/               # Modèles de données (Pydantic)
│   ├── pipelines/            # Pipelines de diffusion
│   │   ├── lcm/                 # Pipelines LCM/LoRA
│   │   └── openvino/            # Pipelines OpenVINO
│   ├── upscaler/             # Modules d'upscaling
│   ├── utils/                # Utilitaires
│   ├── app.py                # Point d'entrée CLI
│   ├── context.py            # Gestion du contexte d'exécution
│   ├── state.py              # État global de l'application
│   └── gguf_diffusion.py     # Wrapper GGUF (stable-diffusion.cpp)
├── models/                # Dossier pour les modèles
│   ├── controlnet_models/
│   ├── lora_models/
│   └── gguf/
│       ├── clip/
│       ├── diffusion/
│       ├── t5xxl/
│       └── vae/
└── scripts/               # Scripts utilitaires
```

### Composants clés

#### 1. **Context** (`context.py`)
Gère le cycle de vie de génération d'images :
- Initialise les pipelines de diffusion
- Mesure la latence
- Gère les erreurs
- Sauvegarde les images

#### 2. **AppSettings** (`app_settings.py`)
Gestion de la configuration :
- Chargement/sauvegarde des paramètres (YAML)
- Liste des modèles disponibles
- Paramètres par défaut

#### 3. **LCMTextToImage** (`pipelines/lcm/text_to_image.py`)
Cœur du système de génération :
- Initialisation des pipelines (LCM, OpenVINO, GGUF)
- Génération d'images (text-to-image, image-to-image)
- Gestion des optimisations (FreeU, Token Merging, TAESD)

#### 4. **State Management** (`state.py`)
Singleton global pour :
- `get_settings()` : Accès aux paramètres
- `get_context()` : Accès au contexte d'exécution
- `get_safety_checker()` : Filtre de contenu NSFW

---

## Modes d'optimisation

### 1. LCM (Latent Consistency Models)

**Principe** : Modèles distillés permettant une génération en 1-4 étapes.

**Avantages** :
- Inférence très rapide (1-2 secondes sur CPU moderne)
- Bonne qualité d'image
- Compatible avec tous les processeurs

**Utilisation** :
```python
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType

settings = get_settings()
context = get_context(InterfaceType.CLI)

# Configuration LCM
settings.settings.lcm_diffusion_setting.lcm_model_id = "stabilityai/sd-turbo"
settings.settings.lcm_diffusion_setting.use_lcm_lora = False
settings.settings.lcm_diffusion_setting.use_openvino = False

# Génération
images = context.generate_text_to_image(settings.settings)
```

**Modèles recommandés** :
- `stabilityai/sd-turbo` (SD 1.5, 512x512)
- `stabilityai/sdxl-turbo` (SDXL, 1024x1024)
- `SimianLuo/LCM_Dreamshaper_v7`

### 2. LCM-LoRA

**Principe** : Adaptateurs LoRA permettant d'utiliser n'importe quel modèle Stable Diffusion avec la vitesse LCM.

**Avantages** :
- Compatible avec des milliers de modèles existants
- Personnalisation facile
- Supporte les fichiers `.safetensors` locaux

**Utilisation** :
```python
settings.settings.lcm_diffusion_setting.use_lcm_lora = True
settings.settings.lcm_diffusion_setting.lcm_lora.base_model_id = "Lykon/dreamshaper-8"
settings.settings.lcm_diffusion_setting.lcm_lora.lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
```

**Modèles de base populaires** :
- `Lykon/dreamshaper-8`
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-xl-base-1.0`

### 3. OpenVINO

**Principe** : Optimisations Intel utilisant le toolkit OpenVINO pour accélérer l'inférence sur CPU Intel et NPU.

**Avantages** :
- Optimisations spécifiques Intel (AVX-512, etc.)
- Support NPU (Neural Processing Unit)
- Modèles pré-compilés pour réduire le temps de démarrage

**Utilisation** :
```python
settings.settings.lcm_diffusion_setting.use_openvino = True
settings.settings.lcm_diffusion_setting.openvino_lcm_model_id = "rupeshs/sd-turbo-openvino"
```

**Configuration NPU** :
```bash
export DEVICE=npu
python -m fastsdcpu.app --use_openvino --prompt "a cat"
```

**Modèles OpenVINO disponibles** :
- `rupeshs/sd-turbo-openvino`
- `rupeshs/sdxl-turbo-openvino`
- Modèles Flux optimisés

### 4. GGUF (via stable-diffusion.cpp)

**Principe** : Modèles quantifiés (Q4_0, Q8_0) pour réduire l'utilisation de la mémoire.

**Avantages** :
- Empreinte mémoire réduite (4-8 bits vs 16-32 bits)
- Pas de dépendance PyTorch lourde
- Idéal pour machines limitées en RAM

**Utilisation** :
```python
settings.settings.lcm_diffusion_setting.use_gguf_model = True
settings.settings.lcm_diffusion_setting.gguf_model.diffusion_path = "models/gguf/diffusion/model.gguf"
settings.settings.lcm_diffusion_setting.gguf_model.clip_path = "models/gguf/clip/clip.gguf"
settings.settings.lcm_diffusion_setting.gguf_model.t5xxl_path = "models/gguf/t5xxl/t5xxl.gguf"
settings.settings.lcm_diffusion_setting.gguf_model.vae_path = "models/gguf/vae/vae.gguf"
```

**Configuration des threads** :
```bash
export GGUF_THREADS=8
```

**Note** : Nécessite la bibliothèque `stable-diffusion.dll/.so/.dylib` dans le dossier `fastsdcpu/`.

---

## Utilisation

### Mode CLI (Command Line Interface)

#### Génération simple

```bash
python -m fastsdcpu.app --prompt "a beautiful sunset over mountains" \
    --inference_steps 4 \
    --image_width 512 \
    --image_height 512
```

#### Avec seed fixe (reproductibilité)

```bash
python -m fastsdcpu.app --prompt "a cat wearing a hat" \
    --seed 42 \
    --number_of_images 4
```

#### Image-to-Image

```bash
python -m fastsdcpu.app --img2img \
    -f input.png \
    --prompt "transform into oil painting" \
    --strength 0.7 \
    --inference_steps 8
```

#### Avec OpenVINO

```bash
python -m fastsdcpu.app --use_openvino \
    --openvino_lcm_model_id "rupeshs/sd-turbo-openvino" \
    --prompt "a futuristic city"
```

#### Avec LCM-LoRA

```bash
python -m fastsdcpu.app --use_lcm_lora \
    --base_model_id "Lykon/dreamshaper-8" \
    --lcm_lora_id "latent-consistency/lcm-lora-sdv1-5" \
    --prompt "a fantasy castle"
```

#### Avec LoRA personnalisé

```bash
python -m fastsdcpu.app --use_lcm_lora \
    --base_model_id "runwayml/stable-diffusion-v1-5" \
    --lora "/path/to/lora.safetensors" \
    --lora_weight 0.8 \
    --prompt "a character in anime style"
```

#### Upscaling

```bash
# EDSR upscaling
python -m fastsdcpu.app --upscale -f input.png

# Tiled SD upscaling (2x)
python -m fastsdcpu.app --sdupscale -f input.png --use_lcm_lora
```

#### Benchmark

```bash
python -m fastsdcpu.app --benchmark \
    --inference_steps 4 \
    --image_width 512 \
    --image_height 512
```

### Mode API Web

#### Démarrer le serveur

```bash
python -m fastsdcpu.app --api --port 8000
```

#### Utilisation de l'API

**Documentation interactive** : `http://localhost:8000/api/docs`

**Endpoints disponibles** :

1. **GET /api/info** - Informations système
```bash
curl http://localhost:8000/api/info
```

2. **GET /api/models** - Liste des modèles disponibles
```bash
curl http://localhost:8000/api/models
```

3. **GET /api/config** - Configuration actuelle
```bash
curl http://localhost:8000/api/config
```

4. **POST /api/generate** - Générer une image
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape",
    "negative_prompt": "blurry, low quality",
    "inference_steps": 4,
    "guidance_scale": 1.0,
    "image_width": 512,
    "image_height": 512,
    "number_of_images": 1,
    "use_openvino": false,
    "use_lcm_lora": false
  }'
```

**Réponse** :
```json
{
  "latency": 1.23,
  "images": ["base64_encoded_image..."],
  "error": ""
}
```

### Mode programmatique (Python)

#### Exemple basique

```python
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType
from fastsdcpu.constants import DEVICE

# Initialisation
settings = get_settings()
context = get_context(InterfaceType.CLI)

# Configuration
config = settings.settings
config.lcm_diffusion_setting.prompt = "a serene mountain landscape"
config.lcm_diffusion_setting.negative_prompt = "blurry, low quality"
config.lcm_diffusion_setting.inference_steps = 4
config.lcm_diffusion_setting.image_width = 512
config.lcm_diffusion_setting.image_height = 512
config.lcm_diffusion_setting.guidance_scale = 1.0
config.lcm_diffusion_setting.use_openvino = False

# Génération
images = context.generate_text_to_image(
    settings=config,
    device=DEVICE
)

# Sauvegarde
if images:
    saved_paths = context.save_images(images, config)
    print(f"Images saved: {saved_paths}")
    print(f"Latency: {context.latency:.2f}s")
```

#### Exemple avec OpenVINO

```python
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType

settings = get_settings()
context = get_context(InterfaceType.CLI)

# Configuration OpenVINO
config = settings.settings
config.lcm_diffusion_setting.use_openvino = True
config.lcm_diffusion_setting.openvino_lcm_model_id = "rupeshs/sd-turbo-openvino"
config.lcm_diffusion_setting.prompt = "a futuristic robot"
config.lcm_diffusion_setting.inference_steps = 2
config.lcm_diffusion_setting.use_tiny_auto_encoder = True

# Génération
images = context.generate_text_to_image(settings=config, device="cpu")
```

#### Exemple Image-to-Image

```python
from PIL import Image
from fastsdcpu.models.lcmdiffusion_setting import DiffusionTask

# Charger l'image d'entrée
init_image = Image.open("input.jpg")

# Configuration
config.lcm_diffusion_setting.diffusion_task = DiffusionTask.image_to_image.value
config.lcm_diffusion_setting.init_image = init_image
config.lcm_diffusion_setting.strength = 0.7  # 0.0 = pas de changement, 1.0 = complètement différent
config.lcm_diffusion_setting.prompt = "transform into watercolor painting"
config.lcm_diffusion_setting.inference_steps = 8

# Génération
images = context.generate_text_to_image(settings=config, device="cpu")
```

#### Exemple avec LoRA

```python
# Configuration LoRA
config.lcm_diffusion_setting.use_lcm_lora = True
config.lcm_diffusion_setting.lcm_lora.base_model_id = "Lykon/dreamshaper-8"
config.lcm_diffusion_setting.lcm_lora.lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

# LoRA personnalisé supplémentaire
config.lcm_diffusion_setting.lora.enabled = True
config.lcm_diffusion_setting.lora.path = "/path/to/custom_lora.safetensors"
config.lcm_diffusion_setting.lora.weight = 0.75

# Génération
images = context.generate_text_to_image(settings=config, device="cpu")
```

#### Génération en lot avec seeds

```python
# Génération avec seed fixe pour reproductibilité
config.lcm_diffusion_setting.use_seed = True
config.lcm_diffusion_setting.seed = 42
config.lcm_diffusion_setting.number_of_images = 4

images = context.generate_text_to_image(settings=config, device="cpu")

# Chaque image aura un seed séquentiel : 42, 43, 44, 45
for img in images:
    print(f"Image seed: {img.info.get('image_seed')}")
```

---

## Configuration avancée

### Fichier de configuration YAML

Le module utilise un fichier `settings.yaml` automatiquement créé dans `configs/`.

**Structure** :
```yaml
lcm_diffusion_setting:
  lcm_model_id: stabilityai/sd-turbo
  openvino_lcm_model_id: rupeshs/sd-turbo-openvino
  prompt: ""
  negative_prompt: ""
  image_height: 512
  image_width: 512
  inference_steps: 1
  guidance_scale: 1.0
  number_of_images: 1
  seed: 123123
  use_seed: false
  use_openvino: false
  use_lcm_lora: false
  use_tiny_auto_encoder: false
  use_safety_checker: false
  clip_skip: 1
  token_merging: 0.0
  
  lcm_lora:
    base_model_id: Lykon/dreamshaper-8
    lcm_lora_id: latent-consistency/lcm-lora-sdv1-5
  
  lora:
    enabled: false
    path: null
    weight: 0.5
    fuse: true

generated_images:
  path: results
  format: PNG
  save_image: true
  save_image_quality: 90
```

### Paramètres JSON personnalisés

```bash
python -m fastsdcpu.app --custom_settings custom_config.json
```

**Exemple `custom_config.json`** :
```json
{
  "prompt": "a beautiful sunset",
  "negative_prompt": "blurry",
  "inference_steps": 6,
  "guidance_scale": 1.5,
  "image_width": 768,
  "image_height": 768,
  "controlnet": {
    "enabled": true,
    "adapter_path": "lllyasviel/control_v11p_sd15_canny",
    "conditioning_scale": 0.5
  }
}
```

### Optimisations disponibles

#### 1. Tiny AutoEncoder (TAESD)

Réduit le temps de décodage VAE (~30% plus rapide).

```python
config.lcm_diffusion_setting.use_tiny_auto_encoder = True
```

**Compatible avec** : SD 1.5, SDXL, Flux

#### 2. Token Merging (ToMe)

Réduit le nombre de tokens traités pour accélérer l'inférence.

```python
config.lcm_diffusion_setting.token_merging = 0.4  # 0.0-1.0
```

**Note** : Valeurs élevées (~0.5+) peuvent réduire la qualité.

#### 3. CLIP Skip

Ignore les dernières couches CLIP pour des effets artistiques.

```python
config.lcm_diffusion_setting.clip_skip = 2  # 1-12
```

**Usage** : Certains modèles sont entraînés avec CLIP Skip (vérifier la fiche du modèle).

#### 4. FreeU

Améliore la qualité des détails (activé automatiquement avec LCM-LoRA).

Améliore les hautes et basses fréquences dans le processus de diffusion.

#### 5. Safety Checker

Filtre les contenus NSFW.

```python
config.lcm_diffusion_setting.use_safety_checker = True
```

**Note** : Ralentit légèrement la génération.

### Variables d'environnement

```bash
# Forcer un device spécifique
export DEVICE=cpu        # cpu, cuda, mps, npu

# Threads GGUF
export GGUF_THREADS=8    # Nombre de threads pour stable-diffusion.cpp
```

---

## API Reference

### Classes principales

#### `Context`

**Emplacement** : `fastsdcpu.context`

**Méthodes** :
- `generate_text_to_image(settings, reshape, device, save_config)` → `List[PIL.Image]`
  - Génère des images selon la configuration
  - `settings` : Instance de `Settings`
  - `reshape` : Recompiler le pipeline OpenVINO (si dimensions changent)
  - `device` : Device cible ("cpu", "cuda", "npu")
  - `save_config` : Sauvegarder la config après génération

- `save_images(images, settings)` → `List[str]`
  - Sauvegarde les images générées
  - Retourne la liste des chemins de fichiers

**Propriétés** :
- `latency` : Temps de génération (secondes)
- `error` : Message d'erreur (vide si succès)

#### `AppSettings`

**Emplacement** : `fastsdcpu.app_settings`

**Méthodes** :
- `load(skip_file)` : Charger la configuration
- `save()` : Sauvegarder la configuration

**Propriétés** :
- `settings` : Instance de `Settings`
- `lcm_models` : Liste des modèles LCM disponibles
- `lcm_lora_models` : Liste des modèles LCM-LoRA
- `openvino_lcm_models` : Liste des modèles OpenVINO
- `stable_diffsuion_models` : Liste des modèles SD standards
- `gguf_diffusion_models` : Liste des modèles GGUF diffusion
- `gguf_clip_models` : Liste des modèles GGUF CLIP
- `gguf_vae_models` : Liste des modèles GGUF VAE

#### `LCMDiffusionSetting`

**Emplacement** : `fastsdcpu.models.lcmdiffusion_setting`

**Champs principaux** :
```python
class LCMDiffusionSetting(BaseModel):
    # Modèles
    lcm_model_id: str
    openvino_lcm_model_id: str
    
    # Prompts
    prompt: str
    negative_prompt: str
    
    # Dimensions
    image_width: int = 512
    image_height: int = 512
    
    # Paramètres de génération
    inference_steps: int = 1
    guidance_scale: float = 1.0
    number_of_images: int = 1
    seed: int = 123123
    use_seed: bool = False
    
    # Modes
    use_openvino: bool = False
    use_lcm_lora: bool = False
    use_tiny_auto_encoder: bool = False
    use_gguf_model: bool = False
    
    # Image-to-Image
    init_image: Any = None
    strength: float = 0.6
    diffusion_task: str = "text_to_image"
    
    # Optimisations
    clip_skip: int = 1
    token_merging: float = 0.0
    use_safety_checker: bool = False
    
    # LoRA
    lcm_lora: LCMLora
    lora: Lora
    
    # ControlNet
    controlnet: Optional[ControlNetSetting] = None
    
    # GGUF
    gguf_model: GGUFModel
```

### Fonctions utilitaires

#### State Management

```python
from fastsdcpu.state import get_settings, get_context, get_safety_checker

# Obtenir les paramètres (singleton)
settings = get_settings(skip_file=False)

# Obtenir le contexte
from fastsdcpu.models.interface_types import InterfaceType
context = get_context(InterfaceType.CLI)

# Obtenir le safety checker
checker = get_safety_checker()
is_safe = checker.is_safe(image)  # PIL.Image → bool
```

#### Chemins

```python
from fastsdcpu.utils.paths import FastStableDiffusionPaths

# Chemins des modèles
lora_path = FastStableDiffusionPaths.get_lora_models_path()
controlnet_path = FastStableDiffusionPaths.get_controlnet_models_path()
gguf_path = FastStableDiffusionPaths.get_gguf_models_path()

# Chemin de résultats
results_path = FastStableDiffusionPaths.get_results_path()

# Chemin pour upscale
upscale_path = FastStableDiffusionPaths.get_upscale_filepath(
    "input.png", scale_factor=2, format="PNG"
)
```

---

## Exemples pratiques

### Exemple 1 : Script de génération batch

```python
"""
Génère plusieurs variations d'un prompt avec différents seeds
"""
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType
from fastsdcpu.constants import DEVICE

def generate_variations(prompt, num_variations=5):
    settings = get_settings()
    context = get_context(InterfaceType.CLI)
    
    config = settings.settings
    config.lcm_diffusion_setting.prompt = prompt
    config.lcm_diffusion_setting.use_seed = False  # Seeds aléatoires
    config.lcm_diffusion_setting.inference_steps = 4
    
    all_images = []
    
    for i in range(num_variations):
        print(f"Generating variation {i+1}/{num_variations}...")
        images = context.generate_text_to_image(config, device=DEVICE)
        
        if images:
            all_images.extend(images)
            saved_paths = context.save_images(images, config)
            print(f"  Saved: {saved_paths[0]}")
            print(f"  Latency: {context.latency:.2f}s")
    
    return all_images

# Utilisation
variations = generate_variations("a cat in a garden", num_variations=5)
print(f"Total images generated: {len(variations)}")
```

### Exemple 2 : Serveur API personnalisé

```python
"""
Serveur API Flask simple pour génération d'images
"""
from flask import Flask, request, jsonify, send_file
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType
from fastsdcpu.constants import DEVICE
from io import BytesIO
import base64

app = Flask(__name__)
settings = get_settings()
context = get_context(InterfaceType.API_SERVER)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    config = settings.settings
    config.lcm_diffusion_setting.prompt = data.get('prompt', '')
    config.lcm_diffusion_setting.negative_prompt = data.get('negative_prompt', '')
    config.lcm_diffusion_setting.inference_steps = data.get('steps', 4)
    config.lcm_diffusion_setting.image_width = data.get('width', 512)
    config.lcm_diffusion_setting.image_height = data.get('height', 512)
    
    images = context.generate_text_to_image(config, device=DEVICE)
    
    if not images:
        return jsonify({'error': context.error}), 500
    
    # Encoder en base64
    buffered = BytesIO()
    images[0].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({
        'image': img_str,
        'latency': context.latency,
        'seed': images[0].info.get('image_seed', -1)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Exemple 3 : Pipeline de traitement d'images

```python
"""
Pipeline : génération → amélioration → upscaling
"""
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType
from fastsdcpu.models.lcmdiffusion_setting import DiffusionTask
from fastsdcpu.upscaler import upscale_image
from fastsdcpu.utils.paths import FastStableDiffusionPaths

def image_pipeline(prompt, enhance_prompt=None, upscale_factor=2):
    settings = get_settings()
    context = get_context(InterfaceType.CLI)
    config = settings.settings
    
    # Étape 1 : Génération initiale
    print("Step 1: Generating base image...")
    config.lcm_diffusion_setting.prompt = prompt
    config.lcm_diffusion_setting.inference_steps = 4
    config.lcm_diffusion_setting.image_width = 512
    config.lcm_diffusion_setting.image_height = 512
    
    images = context.generate_text_to_image(config)
    if not images:
        return None
    
    base_image = images[0]
    print(f"  Generated in {context.latency:.2f}s")
    
    # Étape 2 : Amélioration (si spécifiée)
    if enhance_prompt:
        print("Step 2: Enhancing image...")
        config.lcm_diffusion_setting.diffusion_task = DiffusionTask.image_to_image.value
        config.lcm_diffusion_setting.init_image = base_image
        config.lcm_diffusion_setting.prompt = enhance_prompt
        config.lcm_diffusion_setting.strength = 0.3
        config.lcm_diffusion_setting.inference_steps = 6
        
        enhanced_images = context.generate_text_to_image(config)
        if enhanced_images:
            base_image = enhanced_images[0]
            print(f"  Enhanced in {context.latency:.2f}s")
    
    # Étape 3 : Upscaling
    print(f"Step 3: Upscaling {upscale_factor}x...")
    temp_path = "temp_image.png"
    base_image.save(temp_path)
    
    output_path = FastStableDiffusionPaths.get_upscale_filepath(
        temp_path, upscale_factor, "PNG"
    )
    
    upscaled = upscale_image(context, temp_path, output_path, upscale_factor)
    print(f"  Upscaled and saved to: {output_path}")
    
    return upscaled

# Utilisation
result = image_pipeline(
    prompt="a mystical forest",
    enhance_prompt="add more details and vibrant colors",
    upscale_factor=2
)
```

### Exemple 4 : Comparaison de modèles

```python
"""
Compare plusieurs modèles/configurations pour le même prompt
"""
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType
from fastsdcpu.constants import DEVICE
import matplotlib.pyplot as plt

def compare_models(prompt, configurations):
    """
    configurations: liste de dicts avec les paramètres à tester
    """
    settings = get_settings()
    context = get_context(InterfaceType.CLI)
    
    results = []
    
    for i, config_params in enumerate(configurations):
        print(f"\nTesting configuration {i+1}/{len(configurations)}:")
        print(f"  {config_params['name']}")
        
        config = settings.settings
        config.lcm_diffusion_setting.prompt = prompt
        
        # Appliquer les paramètres
        for key, value in config_params.items():
            if key != 'name':
                setattr(config.lcm_diffusion_setting, key, value)
        
        # Générer
        images = context.generate_text_to_image(config, device=DEVICE)
        
        if images:
            results.append({
                'name': config_params['name'],
                'image': images[0],
                'latency': context.latency
            })
            print(f"  Latency: {context.latency:.2f}s")
    
    return results

# Configuration de test
test_configs = [
    {
        'name': 'LCM Standard',
        'lcm_model_id': 'stabilityai/sd-turbo',
        'use_lcm_lora': False,
        'use_openvino': False,
        'inference_steps': 1
    },
    {
        'name': 'LCM-LoRA Dreamshaper',
        'use_lcm_lora': True,
        'use_openvino': False,
        'inference_steps': 4
    },
    {
        'name': 'OpenVINO',
        'use_openvino': True,
        'use_lcm_lora': False,
        'inference_steps': 2
    }
]

# Comparer
results = compare_models("a serene lake at sunset", test_configs)

# Afficher les résultats
fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
for i, result in enumerate(results):
    axes[i].imshow(result['image'])
    axes[i].set_title(f"{result['name']}\n{result['latency']:.2f}s")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
```

### Exemple 5 : ControlNet (Canny Edge)

```python
"""
Utilisation de ControlNet pour le contrôle de la composition
"""
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType
from fastsdcpu.models.lcmdiffusion_setting import ControlNetSetting
from PIL import Image
import cv2
import numpy as np

def generate_with_controlnet(source_image_path, prompt):
    settings = get_settings()
    context = get_context(InterfaceType.CLI)
    config = settings.settings
    
    # Charger et traiter l'image source
    source_image = Image.open(source_image_path)
    
    # Créer une carte de contours Canny
    image_array = np.array(source_image)
    edges = cv2.Canny(image_array, 100, 200)
    edges_image = Image.fromarray(edges)
    
    # Configuration ControlNet
    controlnet = ControlNetSetting(
        enabled=True,
        adapter_path="lllyasviel/control_v11p_sd15_canny",
        conditioning_scale=0.5
    )
    controlnet._control_image = edges_image
    
    # Configuration de génération
    config.lcm_diffusion_setting.use_lcm_lora = True
    config.lcm_diffusion_setting.controlnet = controlnet
    config.lcm_diffusion_setting.prompt = prompt
    config.lcm_diffusion_setting.inference_steps = 6
    config.lcm_diffusion_setting.guidance_scale = 1.5
    
    # Générer
    images = context.generate_text_to_image(config)
    
    return images[0] if images else None

# Utilisation
result = generate_with_controlnet(
    "reference_photo.jpg",
    "a fantasy castle in anime style"
)
if result:
    result.save("controlnet_result.png")
```

### Exemple 6 : Génération par lots avec queue

```python
"""
Système de queue pour traiter plusieurs prompts
"""
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType
from fastsdcpu.constants import DEVICE
from queue import Queue
from threading import Thread
import time

class ImageGenerationQueue:
    def __init__(self, num_workers=1):
        self.queue = Queue()
        self.results = {}
        self.workers = []
        self.settings = get_settings()
        
        for _ in range(num_workers):
            worker = Thread(target=self._worker)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker(self):
        context = get_context(InterfaceType.CLI)
        config = self.settings.settings
        
        while True:
            item = self.queue.get()
            if item is None:
                break
            
            job_id, params = item
            print(f"Processing job {job_id}...")
            
            # Configurer
            config.lcm_diffusion_setting.prompt = params['prompt']
            config.lcm_diffusion_setting.negative_prompt = params.get('negative_prompt', '')
            config.lcm_diffusion_setting.inference_steps = params.get('steps', 4)
            config.lcm_diffusion_setting.use_seed = True
            config.lcm_diffusion_setting.seed = params.get('seed', int(time.time()))
            
            # Générer
            images = context.generate_text_to_image(config, device=DEVICE)
            
            # Sauvegarder résultat
            if images:
                saved_paths = context.save_images(images, config)
                self.results[job_id] = {
                    'images': images,
                    'paths': saved_paths,
                    'latency': context.latency,
                    'status': 'completed'
                }
            else:
                self.results[job_id] = {
                    'error': context.error,
                    'status': 'failed'
                }
            
            self.queue.task_done()
            print(f"Job {job_id} completed in {context.latency:.2f}s")
    
    def add_job(self, job_id, params):
        self.queue.put((job_id, params))
    
    def wait_completion(self):
        self.queue.join()
    
    def shutdown(self):
        for _ in self.workers:
            self.queue.put(None)
        for worker in self.workers:
            worker.join()

# Utilisation
queue = ImageGenerationQueue(num_workers=2)

# Ajouter des jobs
prompts = [
    "a cyberpunk city at night",
    "a peaceful countryside",
    "an underwater scene with coral",
    "a space station orbiting earth",
    "a medieval market"
]

for i, prompt in enumerate(prompts):
    queue.add_job(f"job_{i}", {'prompt': prompt, 'steps': 4})

# Attendre la fin
print("Waiting for all jobs to complete...")
queue.wait_completion()
queue.shutdown()

# Afficher les résultats
for job_id, result in queue.results.items():
    if result['status'] == 'completed':
        print(f"{job_id}: {result['paths'][0]} ({result['latency']:.2f}s)")
    else:
        print(f"{job_id}: Failed - {result.get('error', 'Unknown error')}")
```

### Exemple 7 : Interpolation entre prompts

```python
"""
Créer une transition fluide entre deux prompts
"""
from fastsdcpu.state import get_settings, get_context
from fastsdcpu.models.interface_types import InterfaceType
from fastsdcpu.constants import DEVICE
from PIL import Image

def interpolate_prompts(prompt_start, prompt_end, num_steps=5):
    """
    Génère des images intermédiaires entre deux prompts
    Note: Cette implémentation simple alterne les seeds
    Pour une vraie interpolation, utiliser des techniques avancées (CLIP embedding interpolation)
    """
    settings = get_settings()
    context = get_context(InterfaceType.CLI)
    config = settings.settings
    
    results = []
    
    # Image de départ
    print(f"Generating start: {prompt_start}")
    config.lcm_diffusion_setting.prompt = prompt_start
    config.lcm_diffusion_setting.use_seed = True
    config.lcm_diffusion_setting.seed = 42
    config.lcm_diffusion_setting.inference_steps = 6
    
    start_images = context.generate_text_to_image(config, device=DEVICE)
    if start_images:
        results.append(start_images[0])
    
    # Images intermédiaires (combinaison de prompts)
    for i in range(1, num_steps - 1):
        ratio = i / (num_steps - 1)
        # Prompt hybride simple
        hybrid_prompt = f"({prompt_start}:0.{int((1-ratio)*10)}) and ({prompt_end}:0.{int(ratio*10)})"
        
        print(f"Generating step {i}/{num_steps-2}: ratio={ratio:.2f}")
        config.lcm_diffusion_setting.prompt = hybrid_prompt
        config.lcm_diffusion_setting.seed = 42 + i
        
        images = context.generate_text_to_image(config, device=DEVICE)
        if images:
            results.append(images[0])
    
    # Image de fin
    print(f"Generating end: {prompt_end}")
    config.lcm_diffusion_setting.prompt = prompt_end
    config.lcm_diffusion_setting.seed = 42 + num_steps
    
    end_images = context.generate_text_to_image(config, device=DEVICE)
    if end_images:
        results.append(end_images[0])
    
    return results

# Utilisation
frames = interpolate_prompts(
    "a sunny day in the park",
    "a starry night in the same park",
    num_steps=5
)

# Créer un GIF
if frames:
    frames[0].save(
        'interpolation.gif',
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0
    )
    print("Animation saved as interpolation.gif")
```

---

## Troubleshooting

### Problèmes courants

#### 1. "Model not found" ou erreurs de téléchargement

**Solution** :
```python
# Utiliser un modèle local
config.lcm_diffusion_setting.use_offline_model = True
config.lcm_diffusion_setting.lcm_model_id = "/path/to/local/model"
```

Ou télécharger manuellement depuis Hugging Face :
```bash
huggingface-cli download stabilityai/sd-turbo --local-dir ./models/sd-turbo
```

#### 2. Out of Memory (OOM)

**Solutions** :
- Réduire la résolution : `image_width=256, image_height=256`
- Utiliser GGUF (quantification) : `use_gguf_model=True`
- Utiliser Tiny AutoEncoder : `use_tiny_auto_encoder=True`
- Réduire batch size : `number_of_images=1`

#### 3. OpenVINO : "Device not found"

**Vérifications** :
```bash
# Vérifier les devices disponibles
python -c "from openvino.runtime import Core; print(Core().available_devices)"
```

**Solutions** :
- Installer OpenVINO correctement
- Utiliser `export DEVICE=cpu` au lieu de `npu`
- Vérifier les drivers Intel

#### 4. GGUF : "Failed to load library"

**Causes** :
- Bibliothèque `stable-diffusion.dll/.so/.dylib` manquante
- Architecture incompatible (ARM vs x86)

**Solutions** :
- Télécharger la bibliothèque compilée pour votre système
- Placer dans le dossier `fastsdcpu/`
- Vérifier les permissions d'exécution (Linux/macOS)

```bash
# Linux/macOS
chmod +x libstable-diffusion.so
```

#### 5. Images floues ou de mauvaise qualité

**Solutions** :
- Augmenter `inference_steps` (4-8 pour LCM)
- Ajuster `guidance_scale` (essayer 1.5-2.0 avec LCM-LoRA)
- Utiliser un meilleur modèle de base (Dreamshaper, etc.)
- Désactiver `use_tiny_auto_encoder`
- Améliorer le prompt (plus de détails)

#### 6. Génération très lente

**Diagnostic** :
```python
# Benchmark pour identifier le goulot d'étranglement
python -m fastsdcpu.app --benchmark
```

**Optimisations** :
- Activer OpenVINO sur processeurs Intel
- Utiliser GGUF pour réduire l'utilisation mémoire
- Activer `use_tiny_auto_encoder=True`
- Activer `token_merging` (0.3-0.5)
- Réduire la résolution
- Augmenter `GGUF_THREADS`

#### 7. Safety Checker bloque tout

**Solution temporaire** :
```python
config.lcm_diffusion_setting.use_safety_checker = False
```

**Note** : Utilisez avec précaution selon votre cas d'usage.

### Logs et debugging

#### Activer les logs détaillés

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Vérifier la configuration chargée

```python
from fastsdcpu.state import get_settings
settings = get_settings()
from pprint import pprint
pprint(settings.settings.lcm_diffusion_setting.model_dump())
```

#### Tester les composants individuellement

```python
# Tester le chargement du pipeline
from fastsdcpu.pipelines.lcm.text_to_image import LCMTextToImage
from fastsdcpu.models.lcmdiffusion_setting import LCMDiffusionSetting

pipeline = LCMTextToImage(device="cpu")
config = LCMDiffusionSetting()
pipeline.init("cpu", config)
print(f"Pipeline loaded: {pipeline.pipeline}")
```

---

## Performance et Benchmarks

### Temps de génération typiques

**Configuration test** : Intel i7-12700K, 32GB RAM, 512x512, 4 steps

| Mode | Modèle | Latence moyenne |
|------|--------|-----------------|
| LCM Standard | sd-turbo | ~1.5s |
| LCM-LoRA | dreamshaper-8 | ~2.0s |
| OpenVINO | sd-turbo-openvino | ~1.2s |
| OpenVINO + TAESD | sd-turbo-openvino | ~0.9s |
| GGUF Q4_0 | flux-schnell | ~3.5s |

### Conseils d'optimisation

1. **Pour la vitesse maximale** :
   - OpenVINO + TAESD sur Intel
   - 1-2 inference steps
   - Résolution 512x512
   - Token merging 0.4

2. **Pour la meilleure qualité** :
   - LCM-LoRA avec modèle de qualité (Dreamshaper)
   - 6-8 inference steps
   - Guidance scale 1.5-2.0
   - Pas de TAESD
   - Résolution native du modèle

3. **Pour économiser la RAM** :
   - GGUF Q4_0/Q8_0
   - TAESD activé
   - Batch size 1
   - Résolution réduite

---

## Roadmap et limitations

### Limitations actuelles

- **GGUF** : Pas de support img2img natif (work in progress)
- **OpenVINO** : LoRA personnalisés non supportés
- **Safety Checker** : Peut avoir des faux positifs
- **ControlNet** : Support partiel selon le pipeline

### Fonctionnalités prévues

- Support Flux complet avec GGUF
- Amélioration de l'interface MCP
- Support des adaptateurs IP-Adapter
- Pipeline vidéo (frame-by-frame)
- Cache de modèles optimisé
- Support AMD ROCm

---

## Contribution

Le projet est open-source. Contributions bienvenues sur :
- GitHub : https://github.com/Flyns157/fastsdcpu
- PyPI : https://pypi.org/project/fastsdcpu-pip/

### Structure de développement

```bash
# Clone
git clone https://github.com/Flyns157/fastsdcpu.git
cd fastsdcpu/pip

# Installation en mode développement
pip install -e .

# Tests
python -m pytest tests/
```

---

## Licence

Le projet utilise une double licence :
- **MIT** pour le code du projet
- **MIT AND (Apache-2.0 OR BSD-2-Clause)** incluant les dépendances tierces

Voir `LICENSE` et `THIRD-PARTY-LICENSES` pour les détails.

---

## Ressources

### Documentation officielle
- Diffusers : https://huggingface.co/docs/diffusers
- OpenVINO : https://docs.openvino.ai
- stable-diffusion.cpp : https://github.com/leejet/stable-diffusion.cpp

### Modèles recommandés
- Hugging Face : https://huggingface.co/models?pipeline_tag=text-to-image
- Civitai : https://civitai.com/

### Communauté
- Issues GitHub : https://github.com/Flyns157/fastsdcpu/issues
- Discussions : https://github.com/Flyns157/fastsdcpu/discussions

---

## Changelog

### v0.1.5 (Actuel)
- Support initial pip
- Pipelines LCM, OpenVINO, GGUF
- API Web et MCP
- Support ControlNet basique

### À venir (v0.2.0)
- Amélioration GGUF img2img
- Cache de pipeline intelligent
- Interface Web améliorée