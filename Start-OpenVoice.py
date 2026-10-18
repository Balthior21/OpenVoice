# pyright: reportMissingImports=false
"""
Lanceur OpenVoice V2 - version avec affichage structuré.

- Texte lu depuis un fichier .txt (ou depuis la variable TEXTE).
- Affichage en étapes séparées par des lignes de #####.
- Barre de progression globale (étapes) et barre de synthèse (phrases).
- Cache de l'empreinte vocale, un cache par fichier de voix.
- Analyse de la voix (Whisper) sur GPU ou processeur au choix.
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
# PARAMÈTRES
# =====================================================================

# Fichier texte à lire (UTF-8), chemin relatif au dossier OpenVoice.
# Laissez "" (vide) pour utiliser la variable TEXTE ci-dessous à la place.
TEXTE_FICHIER = "texte.txt"

# Texte utilisé seulement si TEXTE_FICHIER est vide
TEXTE = "Bonjour, ceci est un test de ma voix clonée avec OpenVoice."

# Langue du texte : "FR", "EN", "ES", "ZH" ou "JP"
LANGUE = "FR"

# Fichier audio de la voix à copier (mp3 ou wav, 10 à 30 secondes, propre).
# Chemin relatif au dossier OpenVoice.
# VOIX_REFERENCE = "resources/Voix-Soph.mp3"
# VOIX_REFERENCE = "resources/example_reference.mp3"
# VOIX_REFERENCE = "resources/Voix-Soph.wav"
VOIX_REFERENCE = "resources/Rec-Soph.mp3"




# Vitesse de parole : 1.0 = normal, 0.8 = plus lent, 1.2 = plus rapide
VITESSE = 1.0

# Seulement si LANGUE = "EN" : "EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"
ACCENT_ANGLAIS = "EN-Default"

# Analyse de la voix de référence (Whisper) :
#   True  = sur le GPU (rapide). Nécessite ctranslate2 version 4.
#   False = sur le processeur (plus lent, mais fonctionne toujours).
# Si le script se ferme brutalement avec une erreur "cudnn ... .dll",
# mettez False.
WHISPER_SUR_GPU = True

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
NB_ETAPES = 5
LARGEUR = 72

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*stft.*")


# ---------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------

def ecrire(texte=""):
    print(texte, flush=True)


def ligne():
    ecrire("#" * LARGEUR)


def titre(texte):
    ecrire()
    ligne()
    ecrire(f"# {texte}")
    ligne()


def barre_etapes(fait, total, largeur=30):
    plein = int(largeur * fait / total)
    return "[" + "█" * plein + "░" * (largeur - plein) + f"] {fait}/{total}"


def debut_etape(numero, nom):
    titre(f"ÉTAPE {numero}/{NB_ETAPES} - {nom}")


def fin_etape(numero, duree):
    ecrire(f"  -> Terminé en {duree:.1f} s")
    ecrire(f"  Progression globale : {barre_etapes(numero, NB_ETAPES)}")


def format_duree(secondes):
    secondes = int(round(secondes))
    minutes, sec = divmod(secondes, 60)
    if minutes:
        return f"{minutes} min {sec:02d} s"
    return f"{sec} s"


# ---------------------------------------------------------------------
# Relance dans le bon environnement
# ---------------------------------------------------------------------

def dans_le_bon_env():
    return Path(sys.prefix).name.lower() == ENV_NAME.lower()


def trouver_python_env():
    """Cherche le Python de l'environnement openvoice sans utiliser le Python global."""
    candidats = []

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
        if candidat.is_file() and candidat.parent.name.lower() == ENV_NAME.lower():
            return candidat
    return None


def relancer_si_necessaire():
    """Relance le script dans l'environnement Conda dédié."""
    if dans_le_bon_env():
        return

    python_env = trouver_python_env()
    if python_env is None:
        ecrire(f"ERREUR : environnement Conda '{ENV_NAME}' introuvable.")
        ecrire("Vérifie le chemin avec : conda env list")
        ecrire("Si besoin, définis OPENVOICE_PYTHON vers le python.exe de l'environnement.")
        sys.exit(1)

    ecrire(f"Relance avec l'environnement OpenVoice : {python_env}")
    resultat = subprocess.call(
        [str(python_env), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(DOSSIER_SCRIPT),
    )
    sys.exit(resultat)


relancer_si_necessaire()
os.chdir(DOSSIER_SCRIPT)


# ---------------------------------------------------------------------
# Configuration et vérifications
# ---------------------------------------------------------------------

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
        ecrire("  Configuration VS Code mise à jour.")
        ecrire("  Si nécessaire : Ctrl+Maj+P > Developer: Reload Window.")
    except OSError as erreur:
        ecrire(f"  Avertissement : configuration VS Code impossible ({erreur})")


def preparer_dll_gpu():
    """Rend visibles les bibliothèques CUDA fournies avec PyTorch
    (utiles à ctranslate2 version 4)."""
    try:
        import torch
        dossier_lib = Path(torch.__file__).resolve().parent / "lib"
        if dossier_lib.exists():
            os.add_dll_directory(str(dossier_lib))
            os.environ["PATH"] = (
                str(dossier_lib) + os.pathsep + os.environ.get("PATH", "")
            )
    except Exception:
        pass


def forcer_whisper_cpu(se_extractor):
    """Force Whisper sur CPU pour contourner les problèmes cuDNN sous Windows."""
    original = se_extractor.WhisperModel

    class WhisperCPU(original):
        def __init__(self, model_size_or_path, *args, **kwargs):
            kwargs.pop("device", None)
            kwargs.pop("compute_type", None)
            super().__init__(
                model_size_or_path, *args, device="cpu", compute_type="int8", **kwargs
            )

    se_extractor.WhisperModel = WhisperCPU


def lire_texte():
    """Lit le texte depuis le fichier .txt, ou depuis la variable TEXTE.
    Retourne (texte, description de la source, nom court)."""
    if TEXTE_FICHIER:
        chemin = Path(TEXTE_FICHIER)
        if not chemin.is_file():
            ecrire(f"ERREUR : fichier texte introuvable : {chemin.resolve()}")
            ecrire("  Créez ce fichier (enregistré en UTF-8), ou videz TEXTE_FICHIER")
            ecrire("  pour utiliser la variable TEXTE du script.")
            sys.exit(1)
        texte = None
        for encodage in ("utf-8-sig", "cp1252"):
            try:
                texte = chemin.read_text(encoding=encodage)
                break
            except UnicodeDecodeError:
                continue
        if texte is None:
            texte = chemin.read_text(encoding="utf-8", errors="replace")
        source, nom = str(chemin), chemin.stem
    else:
        texte, source, nom = TEXTE, "variable TEXTE du script", "texte"

    texte = texte.strip()
    if not texte:
        ecrire("ERREUR : le texte à lire est vide.")
        sys.exit(1)
    return texte, source, nom


def verifier_prerequis():
    """Vérifie les bibliothèques, le GPU, les checkpoints et les données NLTK."""
    requis = ["torch", "librosa", "melo", "openvoice", "nltk", "tqdm"]
    manquants = [nom for nom in requis if importlib.util.find_spec(nom) is None]
    if manquants:
        ecrire(f"ERREUR : bibliothèques manquantes : {', '.join(manquants)}")
        ecrire("Dans le terminal de l'environnement (openvoice) :")
        ecrire("  pip install -e . --no-deps")
        ecrire("  pip install git+https://github.com/myshell-ai/MeloTTS.git")
        sys.exit(1)

    import torch

    if torch.cuda.is_available():
        ecrire(f"  GPU            : {torch.cuda.get_device_name(0)}")
    else:
        ecrire("  ATTENTION : GPU non détecté, le calcul se fera sur le processeur (lent).")
        if "+cpu" in torch.__version__:
            ecrire("  PyTorch installé en version CPU. Pour le GPU :")
            ecrire("    pip uninstall torch torchaudio -y")
            ecrire("    pip install torch==2.8.0 torchaudio==2.8.0 "
                   "--index-url https://download.pytorch.org/whl/cu126")

    if not Path("checkpoints_v2/converter/config.json").is_file():
        ecrire("ERREUR : checkpoints_v2/converter/config.json introuvable.")
        sys.exit(1)
    if not Path("checkpoints_v2/converter/checkpoint.pth").is_file():
        ecrire("ERREUR : checkpoints_v2/converter/checkpoint.pth introuvable.")
        sys.exit(1)
    if not Path(VOIX_REFERENCE).is_file():
        ecrire(f"ERREUR : fichier de voix introuvable : {VOIX_REFERENCE}")
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
                ecrire(f"  Téléchargement des données NLTK : {nom}")
                nltk.download(nom, quiet=True)
    except Exception as erreur:
        ecrire(f"  Avertissement NLTK ignoré : {erreur}")

    if LANGUE == "KR":
        ecrire("ERREUR : le coréen (KR) n'est pas pris en charge sous Windows ici.")
        sys.exit(1)


# ---------------------------------------------------------------------
# Empreinte vocale (avec cache)
# ---------------------------------------------------------------------

def signature_fichier(chemin):
    """Signature légère : chemin, taille et date de modification."""
    chemin = Path(chemin).resolve()
    stat = chemin.stat()
    return {
        "path": str(chemin).lower(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def charger_ou_calculer_empreinte(se_extractor, converter, torch, device, analyse_gpu):
    """
    Réutilise l'empreinte vocale enregistrée (un cache par fichier de voix).
    Elle est recalculée si l'audio de référence ou le checkpoint change.
    """
    reference = Path(VOIX_REFERENCE)
    checkpoint = Path("checkpoints_v2/converter/checkpoint.pth")
    cache_dir = Path(DOSSIER_SORTIE) / "cache"
    cache_file = cache_dir / f"empreinte_{reference.stem}.pth"
    meta_file = cache_dir / f"empreinte_{reference.stem}.json"

    signature = {
        "reference": signature_fichier(reference),
        "checkpoint": signature_fichier(checkpoint),
    }

    if UTILISER_CACHE_VOIX and cache_file.is_file() and meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta == signature:
                try:
                    donnees = torch.load(cache_file, map_location=device, weights_only=True)
                except TypeError:
                    donnees = torch.load(cache_file, map_location=device)
                ecrire("  Empreinte vocale chargée depuis le cache (analyse évitée).")
                return donnees["target_se"]
        except Exception as erreur:
            ecrire(f"  Cache vocal inutilisable, recalcul de l'empreinte ({erreur}).")

    ecrire(f"  Analyse de la voix : {VOIX_REFERENCE}")
    if analyse_gpu:
        ecrire("  (Whisper sur le GPU)")
    else:
        ecrire("  (Whisper sur le processeur : la première analyse peut être longue)")
    empreinte, _ = se_extractor.get_se(str(reference), converter, vad=False)

    if UTILISER_CACHE_VOIX:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"target_se": empreinte.detach().cpu()}, cache_file)
            meta_file.write_text(
                json.dumps(signature, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            ecrire("  Empreinte vocale mise en cache pour les prochaines générations.")
        except Exception as erreur:
            ecrire(f"  Avertissement : impossible d'enregistrer le cache vocal ({erreur}).")

    return empreinte.to(device) if hasattr(empreinte, "to") else empreinte


def duree_audio(chemin):
    """Durée en secondes d'un fichier audio, ou None si illisible."""
    try:
        import soundfile as sf
        return sf.info(str(chemin)).duration
    except Exception:
        return None


# ---------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------

def main():
    debut_total = time.perf_counter()
    durees = {}

    # =================== ÉTAPE 1 : vérifications ====================
    debut_etape(1, "Vérifications et préparation")
    chrono = time.perf_counter()
    configurer_vscode()
    verifier_prerequis()
    texte, source_texte, nom_texte = lire_texte()

    nb_mots = len(texte.split())
    nb_paragraphes = len([p for p in texte.split("\n") if p.strip()])
    ecrire(f"  Voix de réf.   : {VOIX_REFERENCE}")
    ecrire(f"  Langue         : {LANGUE}")
    ecrire(f"  Vitesse        : {VITESSE}")
    ecrire(f"  Texte (source) : {source_texte}")
    ecrire(f"  Texte (taille) : {len(texte)} caractères, {nb_mots} mots, "
           f"{nb_paragraphes} paragraphe(s)")

    import torch
    from tqdm import tqdm
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS

    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    analyse_gpu = WHISPER_SUR_GPU and torch.cuda.is_available()
    if analyse_gpu:
        preparer_dll_gpu()
    else:
        forcer_whisper_cpu(se_extractor)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    Path(DOSSIER_SORTIE).mkdir(parents=True, exist_ok=True)
    durees["Vérifications"] = time.perf_counter() - chrono
    fin_etape(1, durees["Vérifications"])

    # =================== ÉTAPE 2 : convertisseur ====================
    debut_etape(2, "Chargement du convertisseur de voix")
    chrono = time.perf_counter()
    ckpt_converter = Path("checkpoints_v2/converter")
    converter = ToneColorConverter(str(ckpt_converter / "config.json"), device=device)
    converter.load_ckpt(str(ckpt_converter / "checkpoint.pth"))
    durees["Convertisseur"] = time.perf_counter() - chrono
    fin_etape(2, durees["Convertisseur"])

    # =================== ÉTAPE 3 : empreinte vocale =================
    debut_etape(3, "Empreinte de la voix de référence")
    chrono = time.perf_counter()
    target_se = charger_ou_calculer_empreinte(
        se_extractor, converter, torch, device, analyse_gpu
    )
    durees["Empreinte vocale"] = time.perf_counter() - chrono
    fin_etape(3, durees["Empreinte vocale"])

    # =================== ÉTAPE 4 : synthèse vocale ==================
    debut_etape(4, f"Synthèse vocale de base ({LANGUE})")
    chrono = time.perf_counter()
    tts = TTS(language=LANGUE, device=device)
    speakers = tts.hps.data.spk2id
    cle = ACCENT_ANGLAIS if LANGUE == "EN" else LANGUE

    if cle not in speakers:
        ecrire(f"ERREUR : voix '{cle}' inconnue. Voix disponibles : {list(speakers.keys())}")
        sys.exit(1)

    nom_se = cle.lower().replace("_", "-")
    fichier_source_se = Path("checkpoints_v2/base_speakers/ses") / f"{nom_se}.pth"
    if not fichier_source_se.is_file():
        ecrire(f"ERREUR : empreinte du locuteur source introuvable : {fichier_source_se}")
        sys.exit(1)

    horodatage = time.strftime("%Y%m%d_%H%M%S")
    tmp_path = Path(DOSSIER_SORTIE) / f"tmp_{horodatage}_{os.getpid()}.wav"
    sortie = Path(DOSSIER_SORTIE) / f"voix_{nom_texte}_{LANGUE.lower()}_{horodatage}.wav"

    def barre_phrases(elements):
        return tqdm(
            elements,
            desc="  Synthèse",
            unit="phrase",
            file=sys.stdout,
            bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}]",
        )

    try:
        try:
            tts.tts_to_file(texte, speakers[cle], str(tmp_path), speed=VITESSE,
                            pbar=barre_phrases, quiet=True)
        except TypeError:
            # Version de MeloTTS sans options pbar/quiet : barre intégrée.
            tts.tts_to_file(texte, speakers[cle], str(tmp_path), speed=VITESSE)
        durees["Synthèse"] = time.perf_counter() - chrono
        fin_etape(4, durees["Synthèse"])

        # =================== ÉTAPE 5 : conversion ===================
        debut_etape(5, "Conversion vers la voix de référence")
        chrono = time.perf_counter()
        try:
            source_se = torch.load(
                str(fichier_source_se), map_location=device, weights_only=True
            )
        except TypeError:
            source_se = torch.load(str(fichier_source_se), map_location=device)

        ecrire("  Conversion en cours (cette étape n'a pas de barre propre)...")
        converter.convert(
            audio_src_path=str(tmp_path),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(sortie),
            message="@MyShell",
        )
        durees["Conversion"] = time.perf_counter() - chrono
        fin_etape(5, durees["Conversion"])
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError as erreur:
            ecrire(f"  Avertissement : impossible de supprimer le fichier temporaire : {erreur}")

    # =================== RÉSUMÉ FINAL ===============================
    total = time.perf_counter() - debut_total
    longueur_audio = duree_audio(sortie)

    titre("TERMINÉ")
    ecrire(f"  Fichier créé   : {sortie.resolve()}")
    if longueur_audio is not None:
        ecrire(f"  Durée audio    : {format_duree(longueur_audio)}")
    ecrire(f"  Temps total    : {format_duree(total)}")
    ecrire()
    ecrire("  Détail des étapes :")
    for nom, valeur in durees.items():
        ecrire(f"    - {nom:<18} {valeur:6.1f} s")
    ecrire()
    ecrire(f"  Progression globale : {barre_etapes(NB_ETAPES, NB_ETAPES)}")
    ligne()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        ecrire("\nInterrompu par l'utilisateur.")
        sys.exit(130)
