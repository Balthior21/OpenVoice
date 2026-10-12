"""
Lanceur simple pour OpenVoice V2.

Mode d'emploi :
  1. Modifiez les PARAMETRES ci-dessous.
  2. Cliquez sur le bouton ▶ (Run Python File) en haut à droite de VS Code.
  3. Le fichier audio est créé dans le dossier outputs_v2.
"""

import os
import sys
import time
from pathlib import Path

# =====================================================================
#                           PARAMETRES
# =====================================================================

# Le texte que la voix doit dire
TEXTE = "Bonjour, ceci est un test de ma voix clonée avec OpenVoice."

# Langue du texte : "FR", "EN", "ES", "ZH", "JP" ou "KR"
LANGUE = "FR"

# Fichier audio de la voix à copier (mp3 ou wav, 10 à 30 secondes, propre,
# sans musique ni bruit). Chemin relatif au dossier OpenVoice.
VOIX_REFERENCE = "resources/example_reference.mp3"

# Vitesse de parole : 1.0 = normal, 0.8 = plus lent, 1.2 = plus rapide
VITESSE = 1.0

# Seulement si LANGUE = "EN" : "EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"
ACCENT_ANGLAIS = "EN-Default"

# Dossier de sortie
DOSSIER_SORTIE = "outputs_v2"

# =====================================================================
#                  NE RIEN MODIFIER SOUS CETTE LIGNE
# =====================================================================

# On se place toujours dans le dossier du script, où qu'on le lance
os.chdir(Path(__file__).resolve().parent)


def main():
    import torch
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS

    # --- Vérifications de base -------------------------------------
    if not Path(VOIX_REFERENCE).exists():
        print(f"ERREUR : fichier de voix introuvable : {VOIX_REFERENCE}")
        sys.exit(1)
    if not Path("checkpoints_v2/converter/config.json").exists():
        print("ERREUR : dossier checkpoints_v2 introuvable ou incomplet.")
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ATTENTION : GPU non détecté, le calcul se fera sur le processeur (lent).")
    else:
        print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    # --- Chargement du convertisseur de voix -----------------------
    print("Chargement du convertisseur de voix...")
    ckpt_converter = "checkpoints_v2/converter"
    converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
    converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

    # --- Analyse de votre voix de référence ------------------------
    print(f"Analyse de la voix de référence : {VOIX_REFERENCE}")
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