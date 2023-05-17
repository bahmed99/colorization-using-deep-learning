# Coloration d'images

## Dépendances

- Linux ou MacOS (utiliser Firefox si avec MacOS)
- Node (v18.10.0)
- NPM (v8.19.2)
- Python (3.10.6 et 3.9)
- Pip (22.0.2)

Les scripts de lancement installeront les packages node et python nécessaires à l'exécution.

## Initialiser les modèles de colorisation avant de lancer le serveur

Pour pouvoir coloriser des images avec ChromaGAN, il faut télécharger un modèle pré-entraîné (par exemple `ChromaGAN.h5` disponible [ici](https://drive.google.com/drive/folders/12s4rbLmnjW4e8MmESbfRStGbrjOrahlW)), le renommer ChromaGAN.h5 et le placer le dossier `src/models_ai`.

Pour pouvoir coloriser des images avec SIGGRAPH, il faut télécharger les poids du modèle pré-entraîné  (disponible [ici](https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth)) le renommer à SIGGRAPH.pth et le placer le dossier `src/models_ai`.

Pour pouvoir coloriser des images avec ECCV16, il faut télécharger les poids du modèle pré-entraîné  (disponible [ici](https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth)) le renommer à ECCV16.pth et le placer le dossier `src/models_ai`.

Pour pouvoir coloriser des images avec Super Unet V1, il faut télécharger les poids du modèle pré-entraîné  (disponible [ici](https://filesender.renater.fr/?s=download&token=ed15622e-bb47-4eb1-8111-ec889495ee25) vous le trouverez dans le dossier checkpoint/V1_encoder) le renommer à SuperUnetV1.pt et le placer le dossier `src/models_ai`.

Pour pouvoir coloriser des images avec Super Unet V2, il faut télécharger les poids du modèle pré-entraîné  (disponible [ici](https://filesender.renater.fr/?s=download&token=ed15622e-bb47-4eb1-8111-ec889495ee25) vous le trouverez dans le dossier checkpoint/V2_encoder_decoder) le renommer à SuperUnetV2.pt et le placer le dossier `src/models_ai`.

Pour pouvoir coloriser des images avec Unicolor, il faut télécharger les poids des modèles pré-entraînés (y compris Chroma-VQGAN et Hybrid-Transformer) depuis [Hugging Face](https://huggingface.co/luckyhzt/unicolor-pretrained-model/tree/main): 

    - Modèle entraîné avec MSCOCO  - placez le fichier mscoco_step259999.ckpt dans le dossier `src/models_ai` .
    - Deplacer le fichier `config.yaml` dans le dossier `src/models_ai` qui se trouve dans `models/Unicolor/framework/checkpoints/unicolor_mscoco`.
    - Modèle entraîné avec ImageNet - placez le fichier imagenet_step142124.ckpt dans le dossier `framework/checkpoints/unicolor_imagenet`.

Aussi, téléchargez les modèles pré-entraînés (https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/releases/download/v1.0/colorization_checkpoint.zip) à partir [Deep-Exemplar-based-Video-Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization), décompressez le fichier et placez les fichiers dans les dossiers correspondants :

    - `video_moredata_l1` dans le dossier `sample/ImageMatch/checkpoints` 
    - `vgg19_conv.pth` et `vgg19_gray.pth` dans le dossier the `sample/ImageMatch/data` folder


```bash
git clone https://gitlab.com/VillechenaudSimon/colorization-using-optimization src/models_ai/colorization-using-optimization
```

## Conversion d'images couleurs en niveau de gris

Pour lancer la conversion lancer la commande :

```bash
python rgb2gray/rgb2gray.py <directory_colored_img> <directory_gray_img>
```

Le dossier des images en niveau de gris sera crée s'il n'existe pas déjà.  
Des images d'exemple sont disponibles dans rgb2gray/test_color_img et ses résultats sont dans le dossier rgb2gray/res_img.

## Démarrage simple de l'application

Pour simplement lancer l'application, il faut lancer le script suivant (CTRL+C pour stopper) :

```bash
./start_all.sh
```

L'application sera disponible via un navigateur sur le port 3000 : [http://localhost:3000]

## Démarrage séparé

Cette méthode a l'avantage de séparer l'exécution de node et celle de flask dans 2 terminaux différents afin de voir distinctement les logs.

Dans un premier terminal, lancer le serveur avec la commande :

```bash
./run_server.sh
```

Dans un deuxième terminal, lancer le serveur avec la commande :

```bash
./run_client.sh
```

L'application sera disponible via un navigateur sur le port 3000 : [http://localhost:3000]

## Ajouter une méthode de colorisation

Procédure pour ajouter une nouvelle méthode de colorisation :

1. Ajouter un fichier et une classe (nommés ColorizerNOM\_METHODE) héritant directement ou indirectement de la classe models/Colorizer.py.
2. Placer ce fichier dans un sous dossier du dossier models
3. A l'intérieur de la classe, redéfinir, si besoin, la méthode \_\_init\_\_() en prenant soin d'appeler super().\_\_init\_\_().
4. Implémenter la méthode getColorizerName() qui retourne le nom de la méthode (il s'agit de NOM\_METHODE depuis le nom du fichier en .py contenant la classe). A noter que cette valeur sera celle affichée sur l'interface.
5. Implémenter la méthode innerColorize() qui effectue la colorisation conformément à la méthode à ajouter.
6. Placer dans models\_ai les fichiers / repository nécessaire au fonctionnement de cette nouvelle méthode.
7. Relancer le serveur.

## Tests

Des tests sont disponibles, pour vérifier que l'image colorisée obtenue depuis l'interface est la même que celle obtenue depuis le dépôt git de la méthode.
Les tests sont semi-automatisés, c'est-à-dire que l'utilisateur a des choses à faire avant de pouvoir lancer les tests. Il doit s'occuper de déplacer les images qu'il veut comparer dans des dossiers.
Imaginons que l'utilisateur souhaitent comparer 2 images : ocean.jpg et bruschetta.jpg.

Alors, il doit déplacer ocean.jpg et bruscetta.jpg dans un dossier que nous pouvons par exemple appeler "images_interface". (les noms des dossiers sont laissés au choix de l'utilisateur, ils sont donnés ici à titre d'exemple).

Si il souhaite comparer avec les résultats obtenus avec ChromaGAN, il prend ces deux images mais cette fois générée avec le git de Chromagan et il peut les placer dans un dossier nommé "image_chromagan".

Les images dans les deux dossiers doivent alors avoir à peu près le même nom car un tri par ordre alphabétique est utilisé pour les comparer 2 à 2.

Enfin, une fois que l'utilisateur a fait cela, il peut renseigner sa configuration dans le fichier `tst/config.json`.
Dans "methods", il faut ajouter les méthodes que l'ont va comparer ainsi que les dossier où se trouvent les images.
Dans "tests", on peut renseigner les tests que l'on souhaite exécuter. Seules deux fonctions de tests sont disponibles actuellement `proportion_of_different_values` et `SSIM`.
Le "seuil" défini correspond au seuil à partir duquel on considère le test comme réussi ou non. (si le seuil est de 1 pour la SSIM alors on considèrera que le test est réussi si la SSIM des deux images est supérieure ou égale à 1)

## Divers

Il est possible que certains modèles ne parviennent pas à s'exécuter complètement en fonction des performances de la machine. Par exemple, le modèle Unicolor échouait par manque de RAM sur certaines machines avec 8Go de mémoire.

Egalement, si l'application est utilisée avec un Mac, il est possible de devoir modifier les paquets python nécessaires (remplacer tensorflow par tensorflow-macos dans requirements.txt).
