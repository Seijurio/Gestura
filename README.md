<div align="center">

# ✋ Gestura

**Des effets caméra en temps réel que tu cadres et pilotes avec tes mains.**

Fais un cadre avec tes pouces et tes index : l'effet apparaît dedans.
Glisse la main pour changer d'effet, fais 🖖 pour une photo, joue à des mini-jeux, devant ta webcam.

![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-00A98F)
![OpenCV](https://img.shields.io/badge/OpenCV-5-5C3EE8?logo=opencv&logoColor=white)
![60 fps](https://img.shields.io/badge/affichage-60%20i%2Fs-brightgreen)
![Licence MIT](https://img.shields.io/badge/licence-MIT-blue)

</div>

---

## Sommaire

- [Points forts](#points-forts)
- [Démarrage rapide](#démarrage-rapide)
- [Gestes](#gestes)
- [Raccourcis clavier](#raccourcis-clavier)
- [Effets](#effets)
- [Arcade](#arcade)
- [Comment ça marche](#comment-ça-marche)
- [Performances](#performances)
- [Confidentialité](#confidentialité)
- [Personnaliser](#personnaliser)
- [Dépannage](#dépannage)
- [Crédits](#crédits)

## Points forts

- **Le cadre avec les mains** : pouces + index des deux mains forment un quadrilatère où s'applique l'effet, en perspective. Avec une seule main, c'est un rectangle entre le pouce et l'index.
- **33 effets** : rendus photo réalistes (portrait avec bokeh, cinéma, studio, argentique…), effets créatifs (ASCII, matrix, tunnel infini, hologramme, invisibilité…) et un effet plein écran : l'**armure** high-tech.
- **Commandes gestuelles fiables** : poses reconnues en 3D (angles réels des doigts), glissés analysés comme un mouvement complet, poses tenues tolérantes aux hésitations de la détection.
- **Applaudissements au micro** : un claquement sec, et rien d'autre (la voix, le clavier et les sons de l'app sont ignorés).
- **Photo, vidéo et boomerang** enregistrés sans l'interface, avec un aperçu façon polaroïd.
- **Arcade de 5 jeux** : Danse (façon jeu de rythme, suivi du corps), Bulles, Tranche, Jacques a dit, Esquive.
- **Démarrage façon Face ID** : enregistrement du visage et calibration des mains au premier lancement, puis initialisation à chaque ouverture.
- **Lumière d'appoint automatique** : quand il fait sombre, l'écran s'entoure d'un grand cadre blanc lumineux qui t'éclaire (comme une ring light), et l'exposition est relevée. Ce cadre n'apparaît jamais sur les photos.
- **Fluide** : affichage à 60 i/s, découplé de la caméra, et détection en parallèle.

## Démarrage rapide

**Prérequis** : un Mac Apple Silicon, une webcam, Python 3.9 ou plus, et une connexion internet au premier lancement (pour télécharger les modèles).

```bash
git clone https://github.com/Seijurio/Gestura.git
cd Gestura
./run.sh
```

Au premier lancement, `run.sh` :

1. crée l'environnement Python (`.venv`) et installe les dépendances ;
2. télécharge les modèles MediaPipe (≈ 50 Mo) dans `models/` ;
3. lance l'app. macOS demande alors l'accès à la **caméra** et au **micro** pour ton terminal ou VS Code.

| Commande | Effet |
|---|---|
| `./run.sh` | lancement normal (rendu 1080p, affichage 60 i/s) |
| `./run.sh --setup` | refaire la configuration (visage et mains) |
| `./run.sh --4k` | rendu 4K (≈ 30 i/s, pour des photos et vidéos plus grandes) · `--1080p` pour revenir |

**Au premier lancement**, la configuration se fait en quatre étapes : visage (façon Face ID), main ouverte, pince, puis cadre à deux mains. `ESPACE` passe une étape.

## Gestes

| Geste | Action |
|---|---|
| 🤲 Pouces + index des deux mains | le **cadre** où s'applique l'effet |
| 🤏 Une main | rectangle entre le pouce et l'index |
| ✋ Glissé main ouverte, ample et rapide | effet suivant ⬅️ / précédent ➡️ |
| ☝️ Index près d'un bord de l'écran | une flèche ‹ › apparaît, la pointer 0,35 s change d'effet (rester dessus fait défiler) |
| 🖖 Salut vulcain (0,5 s) | **photo** avec compte à rebours |
| 👏👏 Double clap | photo avec compte à rebours |
| ✊ Poing (1 s) | figer ou libérer le cadre (marche pendant le cadrage : le cadre d'avant le geste est conservé) |
| ✋ Main ouverte immobile (1,2 s) | menu circulaire avec aperçu en direct (fermer le poing pour choisir) |
| 👍 / 🤘 / 🫶 | réactions : confettis, lasers, pluie de cœurs |
| 👏 Clap | onde de choc |

> Les gestes de changement d'effet peuvent être verrouillés avec `L`.

## Raccourcis clavier

| Touche | Action | Touche | Action |
|---|---|---|---|
| `1`–`9`, `0` | choisir un effet | `N` / `B` | effet suivant / précédent |
| `[` `]` | taille (pixels, caractères…) | `C` | couleur (ascii, matrix) |
| `M` | masque : l'effet épouse ton visage | `D` | dessin néon dans l'air |
| `Z` / `X` | annuler / effacer le dessin | `G` | capturer le fond (invisibilité) |
| `F` | figer le cadre | `H` | squelette + pose détectée |
| `P` | étincelles au bout des doigts | `L` | verrouiller les gestes |
| `S` | photo | `V` | démarrer / arrêter une vidéo |
| `O` | boomerang | `J` | arcade |
| `E` | plein écran | `TAB` | aide à l'écran |
| `I` | lumière d'appoint : auto, allumée ou éteinte | | |
| `R` | recalibrer | `Q` / `Échap` | quitter |

## Effets

<details open>
<summary><b>Plein écran</b></summary>

| Effet | Description |
|---|---|
| **Armure** | Tu es dans le casque d'une armure high-tech (design original) : démarrage animé, compas qui suit ta tête, verrouillage du visage avec la distance, radar, réacteur lumineux sur la poitrine. Une **main ouverte face à la caméra** charge le propulseur, qui tire ensuite (éclair, onde de choc, son). |

</details>

<details open>
<summary><b>Photo réaliste</b></summary>

| Effet | Description |
|---|---|
| Portrait | arrière-plan flouté avec un vrai bokeh d'objectif |
| Cinéma | étalonnage bleu-orangé, bandes 2.39:1, grain |
| Studio | fond de studio sombre, lumière douce |
| Argentique | noir et blanc de pellicule |
| Vintage | film couleur des années 70 |
| Golden hour | lumière dorée de fin de journée |

</details>

<details>
<summary><b>Créatifs</b></summary>

pixel · flou (verre dépoli) · ascii (vraie police, arêtes orientées) · matrix · tunnel infini · portail temporel (le cadre montre ce qui s'est passé il y a 2 s) · invisibilité (façon Predator) · clones · aura · thermique · neon · glitch · kaléidoscope · bulle · rgb temporel · croquis · halftone · vision nuit · miroir · vhs · pop art · ondulation · hologramme · slit-scan · cartoon · invert

</details>

Et aussi : le **dessin néon** dans l'air (`D`). L'index tendu dessine, la main ouverte gomme, et la barre d'outils se commande du doigt.

## Arcade

`J` ouvre la salle d'arcade : une rangée de grandes cartes colorées. Pointe une carte de l'index (0,8 s, une barre se remplit) ou utilise les touches `1`–`5` et les flèches.

| Jeu | Principe |
|---|---|
| 💃 **Danse** | Des pictogrammes défilent sur une musique synthétisée (3 morceaux). Reproduis la pose avec tes bras pile sur le temps : PARFAIT, SUPER, BIEN ou RATÉ, avec combos, poses dorées ×2 et note de S à D. |
| 🫧 **Bulles** | 40 s pour éclater des bulles du bout des doigts, avec des bonus (ralenti, +5 s, ×2) et des bombes à éviter. |
| 🍉 **Tranche** | Coupe les fruits lancés en l'air d'un geste vif de l'index. Combos multi-fruits, 3 vies. |
| 🙋 **Jacques a dit** | Fais le geste demandé, mais seulement si la consigne commence par « Jacques a dit ». De plus en plus rapide. |
| ☄️ **Esquive** | Ta tête pilote : évite les météores, attrape les étoiles. |

Les records sont sauvegardés localement.

## Comment ça marche

```
 Webcam (thread) ──► image 1080p ──┬──► MediaPipe, en parallèle (LIVE_STREAM)
                                   │      mains · visage · silhouette · corps
                                   │                 │ résultats lissés (filtre One Euro)
                                   ▼                 ▼
                 Boucle d'affichage 60 Hz ◄── reconnaissance des gestes (poses 3D, glissés, claps)
                    │ effet → image propre (photos, vidéos) → interface → fenêtre
                    ▼
                 Micro (thread) ──► détection des claps · Audio (thread) ──► musique et bruitages
```

| Fichier | Rôle |
|---|---|
| `main.py` | point d'entrée, boucle 60 Hz, enchaînement des scènes |
| `tracking.py` | caméra, détection asynchrone MediaPipe, filtres de lissage |
| `gestures.py` | poses (angles 3D des doigts), glissés, poses tenues, claps |
| `live.py` | mode principal : cadre, effets, gestes, photo et vidéo |
| `effects.py` | les effets (une fonction chacun) |
| `armor.py` | effet Armure |
| `ringlight.py` | lumière d'appoint automatique et correction d'exposition |
| `onboarding.py` | configuration (Face ID) et initialisation |
| `games.py` · `arcade.py` | arcade et mini-jeux |
| `audio.py` | mixeur temps réel, synthé, détection des claps |
| `drawing.py` · `menu.py` · `reactions.py` · `icons.py` · `ui.py` | dessin néon, menu circulaire, réactions, icônes, interface |

## Performances

Mesuré sur un Mac Apple Silicon récent, en 1080p :

| | |
|---|---|
| Affichage | 60 i/s, découplé de la caméra (la webcam fournit 30 i/s en 1080p natif) |
| Effets | ≈ 5 ms par image en médiane, 8 ms au maximum |
| Détection des mains | ≈ 7 ms, dans son propre fil |
| Suivi du corps (modèle *heavy*) | ≈ 12 ms, uniquement dans les jeux et effets qui en ont besoin |

Les effets « doux » sont calculés en résolution réduite puis agrandis, sans différence visible. Les effets à motifs (ASCII, matrix, pixel…) restent en résolution native pour garder leur finesse.

## Confidentialité

- **Tout reste sur ta machine.** Aucune image ni aucun son n'est envoyé où que ce soit. La seule connexion sert au téléchargement initial des modèles.
- **Le micro** sert uniquement à mesurer le niveau sonore et à repérer les claps. Rien n'est enregistré.
- **Le déverrouillage par le visage** est un effet visuel, pas une sécurité.
- **Les fichiers personnels sont exclus de git** (`.gitignore`) : `profile.json` (signature du visage), `settings.json` (réglages, records) et `captures/` (photos et vidéos).

## Personnaliser

**Ajouter un effet** : écris une fonction dans `effects.py`, puis ajoute-la à la liste `EFFECTS`.
Pour un effet **privé**, qui ne doit pas être publié, mets-le dans un fichier `local_effects.py` (ignoré par git) qui expose `LOCAL_EFFECTS = [(nom, fonction, besoins)]` : il est chargé automatiquement s'il existe.

```python
def sepia(img, c):
    """Sépia doux."""
    k = np.array([[0.27, 0.53, 0.13], [0.35, 0.69, 0.17], [0.39, 0.77, 0.19]], np.float32)  # BGR
    return cv2.transform(img, k).clip(0, 255).astype(np.uint8)

EFFECTS.append(("sépia", sepia, ()))   # besoins possibles : "seg", "hist", "bg", "face", "pose", "full"
```

**Réglages** (`settings.json`, créé automatiquement) :

| Clé | Rôle |
|---|---|
| `ringlight` | lumière d'appoint : `"auto"`, `"on"` ou `"off"` |
| `render` | `"1080p"` ou `"4k"` |

Les seuils des gestes sont en haut de `gestures.py` (`SWIPE_*`, `HOLD_TIME`).

## Dépannage

| Problème | Solution |
|---|---|
| « Impossible d'ouvrir la caméra » | Réglages Système › Confidentialité et sécurité › Caméra : autorise Terminal ou VS Code, puis relance. |
| Les claps ne marchent pas | Même chose pour le **Micro**. Sans micro, les claps sont détectés à l'image, en moins fiable. |
| Image trop sombre | La lumière d'appoint s'allume toute seule. Passe en plein écran (`E`) pour qu'elle éclaire plus, ou force-la avec `I`. |
| Image floue | Garde le rendu 1080p : la webcam du Mac filme nativement en 1080p, et lui demander de la 4K la fait passer sur un format moins bon. |
| Le glissé change mal d'effet | Fais un geste ample et rapide, main ouverte, ou utilise les flèches ‹ › sur les bords. |
| Danse : les bras décrochent | Recule jusqu'à ce que tes mains levées restent dans l'image (la silhouette guide te place). |

## Crédits

- [MediaPipe](https://developers.google.com/mediapipe) (Google) pour la détection des mains, du visage, du corps et de la silhouette. Les modèles sont téléchargés depuis les serveurs de Google.
- [OpenCV](https://opencv.org), [NumPy](https://numpy.org), [Pillow](https://python-pillow.org) et [sounddevice](https://python-sounddevice.readthedocs.io).
- Les polices et les emojis viennent du système macOS (Helvetica Neue, Avenir Next, Menlo, Apple Color Emoji). Ils ne sont pas inclus dans le dépôt.
- La musique et les bruitages sont synthétisés par le code. Les icônes de l'arcade sont dessinées par le code.

## Licence

[MIT](LICENSE) : libre d'utiliser, modifier et redistribuer, en gardant la mention de copyright.
