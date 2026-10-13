# pyright: reportMissingImports=false
"""
Lanceur OpenVoice V2 optimisé.

Optimisation principale :
- met en cache l'empreinte vocale de la voix de référence ;
- la recalcule automatiquement si le fichier de référence ou le checkpoint change ;
- conserve le moteur OpenVoice / MeloTTS / PyTorch en Python ;
- mesure les durées des principales étapes.

Le cache ne remplace pas les modèles et ne change pas le rendu vocal attendu.
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
# PARAMÈTRES — valeurs originales conservées
# =====================================================================

# Le texte que la voix doit dire
# TEXTE = "Bonjour, ceci est un test de ma voix clonée avec OpenVoice."
TEXTE = """Mon cavalier me fait une cour assidue et convaincante depuis le début de la soirée. Cela n'est pas pour me déplaire. Il doit avoir dix-neuf ans, tout au plus, il est donc de dix ans mon cadet.

J'ai fait sa connaissance par l'intermédiaire de mes voisins de Mobil-Home qui me l'ont recommandé pour l'exécution de menus travaux et autres corvées ménagères. Il gagne ainsi quelque argent de poche pendant les vacances. La première fois que je l'ai prié de venir m'aider, il s'est consacré au lavage de ma voiture.

D'autres fois, il a démoussé le toit de mon Mobil-Home ou lessivé ses parois extérieures ou désherbé mes parterres de fleurs."""

# Langue du texte : "FR", "EN", "ES", "ZH" ou "JP"
LANGUE = "FR"

# Fichier audio de la voix à copier (mp3 ou wav, 10 à 30 secondes, propre).
# Chemin relatif au dossier OpenVoice.
# VOIX_REFERENCE = "resources/Voix-Soph.mp3"
# VOIX_REFERENCE = "resources/example_reference.mp3"
VOIX_REFERENCE = "resources/Voix-Soph.wav"

# Vitesse de parole : 1.0 = normal, 0.8 = plus lent, 1.2 = plus rapide
VITESSE = 1.0

# Seulement si LANGUE = "EN" : "EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"
ACCENT_ANGLAIS = "EN-Default"

# Dossier de sortie
DOSSIER_SORTIE = "outputs_v2"

# Nom de l'environnement conda
ENV_NAME = "openvoice"

# Active/désactive le cache de l'empreinte vocale (recommandé : True).
UTILISER_CACHE_VOIX = True

# =====================================================================
# NE RIEN MODIFIER SOUS CETTE LIGNE
# =====================================================================

DOSSIER_SCRIPT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")


def dans_le_bon_env():
    return Path(sys.prefix).name.lower() == ENV_NAME.lower()


def trouver_python_env():
    """Cherche le Python de l'environnement openvoice sans utiliser le Python global."""
    candidats = []

    # Permet de définir explicitement le chemin si l'installation Conda est atypique.
    python_force = os.environ.get("OPENVOICE_PYTHON")
    if python_force:
        candidats.append(Path(python_force))

    candidats.extend([
        Path.home() / ".conda" / "envs" / ENV_NAME / "python.exe",
        Path(sys.prefix) / "envs" / ENV_NAME / "python.exe",
        Path(sys.base_prefix) / "envs" / ENV_NAME / "python.exe",
        Path(r"G:\Datas\IA\Conda") / "envs" / ENV_NAME / "python.exe",
        Path(r"G:\Datas\IA\MiniConda") / "envs" / ENV_NAME / "python.exe",
        Path(r"G:\Datas\IA\Miniconda3") / "envs" / ENV_NAME / "python.exe",
    ])

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidats.append(Path(conda_prefix) / "python.exe")

    for candidat in candidats:
        if (
            candidat.is_file()
            and candidat.parent.name.lower() == ENV_NAME.lower()
        ):
            return candidat
    return None


def relancer_si_necessaire():
    """Relance le script dans l'environnement Conda dédié."""
    if dans_le_bon_env():
        return

    python_env = trouver_python_env()
    if python_env is None:
        print(f"ERREUR : environnement Conda '{ENV_NAME}' introuvable.")
        print("Vérifie le chemin avec : conda env list")
        print("Si besoin, définis OPENVOICE_PYTHON vers le python.exe de l'environnement.")
        sys.exit(1)

    print(f"Relance avec l'environnement OpenVoice : {python_env}\n")
    resultat = subprocess.call(
        [str(python_env), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(DOSSIER_SCRIPT),
    )
    sys.exit(resultat)


relancer_si_necessaire()
os.chdir(DOSSIER_SCRIPT)


def configurer_vscode():
    """Configure l'interpréteur VS Code en conservant les autres réglages."""
    dossier = DOSSIER_SCRIPT / ".vscode"
    fichier = dossier / "settings.json"
    python_env = str(Path(sys.executable))
    donnees = {}

    if fichier.exists():
        try:
            donnees = json.loads(fichier.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return  # Ne pas toucher à un fichier existant illisible.

    if donnees.get("python.defaultInterpreterPath") == python_env:
        return

    donnees["python.defaultInterpreterPath"] = python_env
    try:
        dossier.mkdir(parents=True, exist_ok=True)
        fichier.write_text(
            json.dumps(donnees, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Configuration VS Code mise à jour.")
        print("Si nécessaire : Ctrl+Maj+P > Developer: Reload Window.\n")
    except OSError as erreur:
        print(f"Avertissement : configuration VS Code impossible ({erreur})")


def forcer_whisper_cpu(se_extractor):
    """Force seulement Whisper sur CPU pour contourner certains problèmes cuDNN Windows."""
    original = se_extractor.WhisperModel

    class WhisperCPU(original):
        def __init__(self, model_size_or_path, *args, **kwargs):
            # Ces options doivent primer sur d'éventuelles valeurs passées par l'appelant.
            kwargs.pop("device", None)
            kwargs.pop("compute_type", None)
            super().__init__(
                model_size_or_path, *args, device="cpu", compute_type="int8", **kwargs
            )

    se_extractor.WhisperModel = WhisperCPU


def verifier_prerequis():
    """Vérifie les bibliothèques, le GPU, les checkpoints et les données NLTK."""
    requis = ["torch", "librosa", "melo", "openvoice", "nltk"]
    manquants = [
        nom for nom in requis
        if importlib.util.find_spec(nom) is None
    ]
    if manquants:
        print(f"ERREUR : bibliothèques manquantes : {', '.join(manquants)}")
        print("Dans le terminal de l'environnement (openvoice) :")
        print("  pip install -e . --no-deps")
        print("  pip install git+https://github.com/myshell-ai/MeloTTS.git")
        sys.exit(1)

    import torch

    if torch.cuda.is_available():
        print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")
    else:
        print("ATTENTION : GPU non détecté, le calcul se fera sur le processeur (lent).")
        if "+cpu" in torch.__version__:
            print("PyTorch installé en version CPU. Vérifie l'installation CUDA de cet environnement.")

    if not Path("checkpoints_v2/converter/config.json").is_file():
        print("ERREUR : checkpoints_v2/converter/config.json introuvable.")
        sys.exit(1)
    if not Path("checkpoints_v2/converter/checkpoint.pth").is_file():
        print("ERREUR : checkpoints_v2/converter/checkpoint.pth introuvable.")
        sys.exit(1)
    if not Path(VOIX_REFERENCE).is_file():
        print(f"ERREUR : fichier de voix introuvable : {VOIX_REFERENCE}")
        sys.exit(1)

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
    except Exception as erreur:
        print(f"Avertissement NLTK ignoré : {erreur}")

    if LANGUE == "KR":
        print("ERREUR : le coréen (KR) n'est pas pris en charge sous Windows ici.")
        sys.exit(1)


def signature_fichier(chemin):
    """Signature légère : chemin, taille et date de modification."""
    chemin = Path(chemin).resolve()
    stat = chemin.stat()
    return {
        "path": str(chemin).lower(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def charger_ou_calculer_empreinte(se_extractor, converter, torch, device):
    """
    Réutilise l'empreinte vocale enregistrée.
    Elle est recalculée si l'audio de référence ou le checkpoint change.
    """
    reference = Path(VOIX_REFERENCE)
    checkpoint = Path("checkpoints_v2/converter/checkpoint.pth")
    cache_dir = Path(DOSSIER_SORTIE) / "cache"
    cache_file = cache_dir / "empreinte_voix_reference.pth"
    meta_file = cache_dir / "empreinte_voix_reference.json"

    signature = {
        "reference": signature_fichier(reference),
        "checkpoint": signature_fichier(checkpoint),
    }

    if UTILISER_CACHE_VOIX and cache_file.is_file() and meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta == signature:
                # weights_only est disponible dans les versions récentes de PyTorch.
                try:
                    donnees = torch.load(cache_file, map_location=device, weights_only=True)
                except TypeError:
                    donnees = torch.load(cache_file, map_location=device)
                empreinte = donnees["target_se"]
                print("Empreinte vocale chargée depuis le cache (analyse Whisper évitée).")
                return empreinte
        except Exception as erreur:
            print(f"Cache vocal inutilisable, recalcul de l'empreinte ({erreur}).")

    print(f"Analyse de la voix de référence : {VOIX_REFERENCE}")
    print("(Whisper fonctionne sur le processeur ; première analyse potentiellement longue.)")
    empreinte, _ = se_extractor.get_se(str(reference), converter, vad=False)

    if UTILISER_CACHE_VOIX:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"target_se": empreinte.detach().cpu()}, cache_file)
            meta_file.write_text(
                json.dumps(signature, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("Empreinte vocale mise en cache pour les prochaines générations.")
        except Exception as erreur:
            print(f"Avertissement : impossible d'enregistrer le cache vocal ({erreur}).")

    return empreinte.to(device) if hasattr(empreinte, "to") else empreinte


def main():
    debut_total = time.perf_counter()
    configurer_vscode()
    verifier_prerequis()

    import torch
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS

    forcer_whisper_cpu(se_extractor)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    Path(DOSSIER_SORTIE).mkdir(parents=True, exist_ok=True)

    # Chargement du convertisseur de voix
    chrono = time.perf_counter()
    print("Chargement du convertisseur de voix...")
    ckpt_converter = Path("checkpoints_v2/converter")
    converter = ToneColorConverter(str(ckpt_converter / "config.json"), device=device)
    converter.load_ckpt(str(ckpt_converter / "checkpoint.pth"))
    print(f"Convertisseur chargé en {time.perf_counter() - chrono:.1f} s.")

    # Analyse ou chargement de l'empreinte de référence (mise en cache)
    chrono = time.perf_counter()
    target_se = charger_ou_calculer_empreinte(se_extractor, converter, torch, device)
    print(f"Empreinte prête en {time.perf_counter() - chrono:.1f} s.")

    # Synthèse de la voix de base
    chrono = time.perf_counter()
    print(f"Génération de la voix de base ({LANGUE})...")
    tts = TTS(language=LANGUE, device=device)
    speakers = tts.hps.data.spk2id
    cle = ACCENT_ANGLAIS if LANGUE == "EN" else LANGUE

    if cle not in speakers:
        print(f"ERREUR : voix '{cle}' inconnue. Voix disponibles : {list(speakers.keys())}")
        sys.exit(1)

    nom_se = cle.lower().replace("_", "-")
    fichier_source_se = Path("checkpoints_v2/base_speakers/ses") / f"{nom_se}.pth"
    if not fichier_source_se.is_file():
        print(f"ERREUR : empreinte du locuteur source introuvable : {fichier_source_se}")
        sys.exit(1)

    horodatage = time.strftime("%Y%m%d_%H%M%S")
    tmp_path = Path(DOSSIER_SORTIE) / f"tmp_{horodatage}_{os.getpid()}.wav"
    sortie = Path(DOSSIER_SORTIE) / f"voix_{LANGUE.lower()}_{horodatage}.wav"

    try:
        tts.tts_to_file(TEXTE, speakers[cle], str(tmp_path), speed=VITESSE)
        print(f"Synthèse terminée en {time.perf_counter() - chrono:.1f} s.")

        chrono = time.perf_counter()
        try:
            source_se = torch.load(
                str(fichier_source_se), map_location=device, weights_only=True
            )
        except TypeError:
            source_se = torch.load(str(fichier_source_se), map_location=device)

        print("Conversion vers la voix de référence...")
        converter.convert(
            audio_src_path=str(tmp_path),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(sortie),
            message="@MyShell",
        )
        print(f"Conversion terminée en {time.perf_counter() - chrono:.1f} s.")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError as erreur:
            print(f"Avertissement : impossible de supprimer le fichier temporaire : {erreur}")

    print("\nTERMINÉ !")
    print(f"Fichier créé : {sortie.resolve()}")
    print(f"Durée totale : {time.perf_counter() - debut_total:.1f} s.")


if __name__ == "__main__":
    main()
