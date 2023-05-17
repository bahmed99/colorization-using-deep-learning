# Cas d'utilisation et résultats avec la dataset div2k

## Cas d'utilisation

### Téléchargement de la dataset

```bash
$ cd rgb2gray
$ wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
$ wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
$ mkdir div2k
$ unzip DIV2K_train_LR_bicubic_X4.zip -d div2k/
$ unzip DIV2K_valid_LR_bicubic_X4.zip -d div2k/
$ rm *.zip
$ mv div2k/DIV2K_train_LR_bicubic/X4/* div2k/
$ mv div2k/DIV2K_valid_LR_bicubic/X4/* div2k/
$ rm -r -d div2k/DIV2K_valid_LR_bicubic/
$ rm -r -d div2k/DIV2K_train_LR_bicubic/

```

### Conversion en niveau de gris

Pour lancer la conversion lancer la commande :

```bash
python rgb2gray.py div2k ../src/view/public/uploads/Datasets/div2kGray
```

### Colorisation via l'interface

Aller dans colorize > Choose an existing dataset > div2kGray.
Et choisir les methodes de colorisation voulues.


## Résultats

- script ChromaGAN : **8min56**
- Colorisation ChromaGan via interface : **3min10**
- Colorisation ECCV16 via interface : **5min13**
- Colorisation SIGGRAPH via interface : **13min25**
- 🠮 Temps total pour les 3 methodes  : **21min48**
- Colorisation avec les 3 methodes via interface : **22min27**
- 🠮 Différence négligeable
