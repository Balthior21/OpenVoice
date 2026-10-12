# pyright: reportMissingImports=false
"""
Lanceur OpenVoice V2 (Start-OpenVoice.py)

- Se relance automatiquement dans l'environnement conda "openvoice".
- Configure VS Code pour utiliser ce bon environnement (une seule fois).
- Vérifie les prérequis (bibliothèques, GPU, données NLTK).
- Génère un fichier audio dans le dossier outputs_v2.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# =====================================================================
#                           PARAMETRES
# =====================================================================

# Le texte que la voix doit dire
#TEXTE = "Bonjour, ceci est un test de ma voix clonée avec OpenVoice."
TEXTE = """  Mon cavalier me fait une cour assidue et convaincante depuis le début de la soirée. Cela n'est pas pour me déplaire. Il doit avoir dix-neuf ans, tout au plus, il est donc de dix ans mon cadet.
J'ai fait sa connaissance par l'intermédiaire de mes voisins de Mobil-Home qui me l'ont recommandé pour l'exécution de menus travaux et autres corvées ménagères. Il gagne ainsi quelque argent de poche pendant les vacances. La première fois que je l'ai prié de venir m'aider, il s'est consacré au lavage de ma voiture. 
D'autres fois, il a démoussé le toit de mon Mobil-Home ou lessivé ses parois extérieures ou désherbé mes parterres de fleurs."""


# Langue du texte : "FR", "EN", "ES", "ZH" ou "JP"
LANGUE = "FR"

# Fichier audio de la voix à copier (mp3 ou wav, 10 à 30 secondes, propre,
# sans musique ni bruit). Chemin relatif au dossier OpenVoice.
#VOIX_REFERENCE = "resources/Voix-Soph.mp3"
#VOIX_REFERENCE = "resources/example_reference.mp3"
VOIX_REFERENCE = "resources/Voix-Soph.wav"

# Vitesse de parole : 1.0 = normal, 0.8 = plus lent, 1.2 = plus rapide
VITESSE = 1.0

# Seulement si LANGUE = "EN" : "EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"
ACCENT_ANGLAIS = "EN-Default"

# Dossier de sortie
DOSSIER_SORTIE = "outputs_v2"

# Nom de l'environnement conda
ENV_NAME = "openvoice"

# =====================================================================
#                  NE RIEN MODIFIER SOUS CETTE LIGNE
# =====================================================================

DOSSIER_SCRIPT = Path(__file__).resolve().parent

# Réduit le bruit dans le terminal
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")


def dans_le_bon_env():
    return Path(sys.prefix).name.lower() == ENV_NAME.lower()


def trouver_python_env():
    """Cherche le python.exe de l'environnement openvoice."""
    candidats = [
        Path.home() / ".conda" / "envs" / ENV_NAME / "python.exe",
        Path(sys.prefix) / "envs" / ENV_NAME / "python.exe",
        Path(sys.base_prefix) / "envs" / ENV_NAME / "python.exe",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidats.append(Path(conda_prefix) / "python.exe")
    for c in candidats:
        if c.exists() and c.parent.name.lower() == ENV_NAME.lower():
            return c
    return None


def relancer_si_necessaire():
    """Si on n'est pas dans l'environnement openvoice, on s'y relance."""
    if dans_le_bon_env():
        return
    python_env = trouver_python_env()
    if python_env is None:
        print(f"ERREUR : environnement conda '{ENV_NAME}' introuvable.")
        print("Créez-le avec : conda create -n openvoice python=3.9 -y")
        sys.exit(1)
    print(f"Mauvais Python détecté, relance avec : {python_env}\n")
    code = subprocess.call(
        [str(python_env), str(Path(__file__).resolve())] + sys.argv[1:]
    )
    sys.exit(code)


relancer_si_necessaire()

# À partir d'ici, on est sûr d'être dans l'environnement openvoice
os.chdir(DOSSIER_SCRIPT)


def configurer_vscode():
    """Indique à VS Code le bon Python pour ce dossier (.vscode/settings.json).
    Les réglages déjà présents sont conservés."""
    dossier = DOSSIER_SCRIPT / ".vscode"
    fichier = dossier / "settings.json"
    python_env = str(Path(sys.executable))
    donnees = {}
    if fichier.exists():
        try:
            donnees = json.loads(fichier.read_text(encoding="utf-8"))
        except Exception:
            return  # fichier existant illisible : on n'y touche pas
    if donnees.get("python.defaultInterpreterPath") == python_env:
        return
    donnees["python.defaultInterpreterPath"] = python_env
    try:
        dossier.mkdir(exist_ok=True)
        fichier.write_text(json.dumps(donnees, indent=4), encoding="utf-8")
        print("Configuration VS Code mise à jour.")
        print("  Pour supprimer les soulignements : Ctrl+Maj+P, puis "
              "'Developer: Reload Window' (une seule fois).\n")
    except Exception as e:
        print(f"Avertissement : configuration VS Code impossible ({e})")


def forcer_whisper_cpu(se_extractor):
    """L'analyse de la voix de référence (Whisper) tourne sur le processeur.
    Évite l'erreur 'cudnn_ops_infer64_8.dll' sous Windows. Le reste
    (génération et conversion de la voix) reste sur le GPU."""
    original = se_extractor.WhisperModel

    class WhisperCPU(original):
        def __init__(self, model_size_or_path, *args, **kwargs):
            super().__init__(model_size_or_path, device="cpu", compute_type="int8")

    se_extractor.WhisperModel = WhisperCPU


def verifier_prerequis():
    """Vérifie bibliothèques, GPU, modèles et données NLTK."""
    # 1. Bibliothèques
    requis = ["torch", "librosa", "melo", "openvoice", "nltk"]
    manquants = [m for m in requis if importlib.util.find_spec(m) is None]
    if manquants:
        print(f"ERREUR : bibliothèques manquantes : {', '.join(manquants)}")
        print("Dans le terminal (avec (openvoice) visible), installez-les :")
        print("  pip install -e . --no-deps")
        print("  pip install git+https://github.com/myshell-ai/MeloTTS.git")
        sys.exit(1)

    import torch

    # 2. GPU
    if torch.cuda.is_available():
        print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")
    else:
        print("ATTENTION : GPU non détecté, le calcul se fera sur le processeur (lent).")
        if "+cpu" in torch.__version__:
            print("  PyTorch est en version CPU. Pour le GPU :")
            print("  pip uninstall torch torchaudio -y")
            print("  pip install torch==2.8.0 torchaudio==2.8.0 "
                  "--index-url https://download.pytorch.org/whl/cu126")

    # 3. Modèles et voix de référence
    if not Path("checkpoints_v2/converter/config.json").exists():
        print("ERREUR : dossier checkpoints_v2 introuvable ou incomplet.")
        sys.exit(1)
    if not Path(VOIX_REFERENCE).exists():
        print(f"ERREUR : fichier de voix introuvable : {VOIX_REFERENCE}")
        sys.exit(1)

    # 4. Données NLTK (téléchargées automatiquement si absentes)
    try:
        import nltk
        for chemin, nom in [
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("corpora/cmudict", "cmudict"),
        ]:
            try:
                nltk.data.find(chemin)
            except LookupError:
                print(f"Téléchargement des données NLTK : {nom}")
                nltk.download(nom, quiet=True)
    except Exception as e:
        print(f"Avertissement NLTK ignoré : {e}")

    if LANGUE == "KR":
        print("ERREUR : le coréen (KR) n'est pas pris en charge sous Windows ici.")
        sys.exit(1)


def main():
    configurer_vscode()
    verifier_prerequis()

    import torch
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS

    forcer_whisper_cpu(se_extractor)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    # --- Chargement du convertisseur de voix -----------------------
    print("Chargement du convertisseur de voix...")
    ckpt_converter = "checkpoints_v2/converter"
    converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
    converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

    # --- Analyse de votre voix de référence ------------------------
    print(f"Analyse de la voix de référence : {VOIX_REFERENCE}")
    print("  (étape sur processeur : quelques dizaines de secondes la 1re fois)")
    target_se, _ = se_extractor.get_se(VOIX_REFERENCE, converter, vad=False)

    # --- Synthèse de la voix de base (MeloTTS) ---------------------
    print(f"Génération de la voix de base ({LANGUE})...")
    tts = TTS(language=LANGUE, device=device)
    speakers = tts.hps.data.spk2id

    cle = ACCENT_ANGLAIS if LANGUE == "EN" else LANGUE
    if cle not in speakers:
        print(f"ERREUR : voix '{cle}' inconnue. Voix disponibles : {list(speakers.keys())}")
        sys.exit(1)

    tmp_path = f"{DOSSIER_SORTIE}/tmp.wav"
    tts.tts_to_file(TEXTE, speakers[cle], tmp_path, speed=VITESSE)

    # --- Conversion vers votre voix --------------------------------
    nom_se = cle.lower().replace("_", "-")
    source_se = torch.load(
        f"checkpoints_v2/base_speakers/ses/{nom_se}.pth", map_location=device
    )

    horodatage = time.strftime("%Y%m%d_%H%M%S")
    sortie = f"{DOSSIER_SORTIE}/voix_{LANGUE.lower()}_{horodatage}.wav"

    print("Conversion vers la voix de référence...")
    converter.convert(
        audio_src_path=tmp_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=sortie,
        message="@MyShell",
    )

    os.remove(tmp_path)
    print("\nTERMINE !")
    print(f"Fichier créé : {Path(sortie).resolve()}")


if __name__ == "__main__":
    main()