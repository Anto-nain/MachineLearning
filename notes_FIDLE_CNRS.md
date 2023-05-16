


# Sequence 1 : Concepts et bases des réseau de neurones profonds (DNN)

## Descente de gradient

A partir d'un nuage de points, on définit une grandeur $\theta$ qui comporte les paramètres de la courbe (polynomiale) que l'on souhaite calquer sur notre modèle. Ainsi, on cherche à obtenir  $y = f(x)$ avec f polynomiale (souvent linéaire).
$$P(x) = \theta_0 + \theta_1x + \theta_2x^2 + ... + \theta_nx^n$$
$$\theta = [\theta_0, \theta_1, \theta_2, ..., \theta_n]$$
Par la suite, on défini une fonction de coût $J(\theta)$ qui permet de mesurer l'erreur entre nos points du nuage expérimentaux et ceux obtenus par notre modèle $P(x)$. Il existe plusieurs façon de calculer cette erreur : 
  * L'erreur quadratique moyenne (RMSE)
  * La moyenne du carré des erreurs (MSE)

Pour finir, on applique l'algorithme de descente du grandient pour trouver les paramètres $\theta$ qui minimisent la fonction de coût $J(\theta)$ :

$$\theta \leftarrow \theta - \alpha \frac{\partial}{\partial \theta}J(\theta)$$
Avec $\alpha$ le taux d'apprentissage.

**Remarques** : 
* On peut noter que $x$ peut être un vecteur : plusieurs points en entrée. Et la sortie $y$ peut aussi en être un vecteur dont chaque composante vaut $y_i := P_i(x)$ (soit chaque $y_i$ est polynomiale de $x$). 
* Il faut faire attention lorsque l'on choisit le taux d'apprentissage $\alpha$ Pour avoir une convergence suffisamment rapide sans pour autant faire du sur-apprentissage : **Toujours vérifier que l'on ne fait ni du sur-apprentissage ni du sous-apprentissage**.

## Régression linéaire et logistique

Grâce à la descente du gradient on peut directement résoudre des problèmes de régressions linéaire : on prend simplement $P(x) = \theta_0 + \theta_1x$. Si on veut que ça soit polynomial, bien sûr on rajoutera des termes dans $P$.  
Pour La régression logistique, on part d'une régression linéaire avec $t$ la sortie linéaire de X (vecteur définissant un point du nuage) : $t = W \cdot X + b$. Ensuite, on rajoute une fonction d'activation $\sigma$ qui prend la valeur de $t$ et la met entre 0 et 1. On est donc dans le cas d'une régression logistique : $y := \sigma(t) := \sigma(W \cdot X + b)$. Ici, la descente du grandient va se faire sur $W$ la matrice des poids et $b$ le biais (les deux étant équivalent à $\theta$ pour une régression linéaire).

**En résumé :**

|  	| Régression linéaire 	| Régression logistique 	|  	|  	|
|---	|---	|---	|---	|---	|
| Paramètre à optimiser par descente du gradient 	| $\theta$ 	| $W$, $b$ 	|  	|  	|
| Formule 	| $y = \theta_1x + \theta_0$	| $y = \sigma(W.X + b)$      	|  	|  	|

**Attention :**  
Dans le cas générale, on utilise jamais de régression polynomiale dans un neurone. On préfère utiliser plusieurs neurones (de forme linéaire) et les connnecter entre eux. La question est maintenant d'optimiser les poids et biais de tout les neurones.

## Réseau de neurones profonds
### Fonctionnement

 Un réseau de neurone se présente la façon suivant :
 * En entrée, on a X un vecteur représentant un point de notre nuage de points.
 * En sortie on a Y un vecteur (ou scalaire) représentant les prédictions de notre modèle. Ici, Y peut possèder des valeurs umériques (type régression linéaire) ou des valeurs en 0 et 1 (type régression logistique).
 * Entre les deux, il yu a un certain nombre de neurones interconnnectés rangée par profondeur (couche 1, couche 2, ...).
 * Chaque neurone prend en entrée la sortie de tout les neurones de la couche précédente. Pour chaque entrée il associe une poid ($w$ contenu dans $W$) et un biais global ($b$). Il applique ensuite une fonction d'activation $\sigma$ pour obtenir sa sotie.
### Obtention des poids et biais
* Pour tout les points $X^i$ du nuage, on récupère ${\hat{Y}}^i$ prédit via le réseau de neurone
* On compare ${\hat{Y}}^i$ prédit avec $Y^i$ du nuage pour obtenir l'erreur $E_{W,b}(Y^i, {\hat{Y}}^i)$
* Avec un algorithme de rétropropagation de l'erreur, on corrige les poids ($W$) et biais ($b$) pour chaque neurone afin de minimiser l'erreur $E_{W,b}(Y^i, {\hat{Y}}^i)$. 

On répète cette opération plusieurs fois jusqu'à ce que l'erreur soit suffisamment faible.  
**Remarques :**  
Dans la rétropropagation de l'erreur on retrouve une forme de descente du gradient avec un taux d'apprentissage, donc là aussi il faut faire attention à ne pas faire du sur-apprentissage ou du sous-apprentissage.

## Entrainer un réseau de neurones

En partant de notre set de données, on va diviser notre set en 2 parties :
* Un set d'entrainement (80%) sur lequel on va entrainer nos neurones pour minimiser l'erreur.
* Un set de test (20%) sur lequel on va tester notre modèle (au cours de son apprentissage) pour visualiser l'évolution de l'erreur sur un set indépendant de l'apprentissage. Cela nous permet nottamment de visualiser le sous-apprentissage et le sur-apprentissage. 

**Attention :**   
Il faut avoir des données normalisés séparement pour le set d'entrainement et le set de test (car les deux sont indépendants).  
Pour faire cela, on va normaliser nos données d'entrée $X_train$ et $X_test$ (mais pas $Y_train$ et $Y_test$) de la façon suivante : 

$$X_{train\_norm} = \frac{X_{train} - \mu_{train}}{\sigma_{train}}$$

avec $\mu_{train}$ la moyenne de $X_{train}$ (`.mean() `sous python ) et $\sigma_{train}$ l'écart-type de $X_{train}$ (`.std()` sous python).

## Différentes sorties de notre réseau de neurones

### Problème de régression 

On cherche à obtenir un ou plusieurs arguments en sortie de notre réseau de neurones. La taille de $Y$ vaut ce nombre d'arguments. Dans le cadre d'un problème de régression linéaire, les valurs dans $Y$ sont des nombres réels. C'est ce qui est naturellement sorti par notre réseau de neurones.

### Problème de classification

En sortie de notre réseau, on cherche à savoir si notre point d'entrée appartient à telle ou telle **classe**. Ainsi, en fin de notre réseau de neurones, on va transformer la sortie en $-\inf$ et $+\inf$ en une valeur entre 0 et 1 (soit une probabilité d'appartenance à une classe) en utilisant une fonction d'activation ($Sigmoide$ ou bien $SoftMax$). Finalement, on va choisir la classe qui a la plus grande probabilité d'appartenance en utilisant une fonction de décision (une autre fonction d'activation) : $ArgMax$.  
Seule la Première fonction d'activation est mise dans le réseau de neurones. La fonction coût ne sera ici plus la même que celle utilisée en régression linéaire : on utilisera la **fonction coût de la régression logistique à base d'entropie croisée**.

Pour bien interprèter le résultat, on peut dresser un tableau de prédiction : (vraies classes // classes prédites) et remarquer si certaines classes on plus tendances à être mal prédites ou confondues que d'autres.

## Conclusion
Avec la méthode présentée ici, on peut construire des réseaux de neurones profonds : **DNN** (*Deep Neural Network*).


# Petit point sur les fonctions d'activation

 **A noter :** 
 Une fonction d'activation doit être dérivable au sens numérique afin d'utiliser la descente du gradient.

* Au sein de notre réseau de neurone, la fonction d'activation la plus utilisée est **ReLU** (Rectified Linear Unit) : $f(x) = max(0, x)$.
* En couche de sortie de notre réseau de neurone, trois choix s'offrent à nous :
  * Si on réalise une régression, il n'y a besoin de fonction d'activation particulière et on récupère directement le résultat en sortie du neurone.
  * Si on réalise une classification binaire (True // False ou 1 // 0), on utilise majoritairement la fonction d'activation **Sigmoide** : $f(x) = \frac{1}{1 + e^{-x}}$.
  * Si on réalise une classification à plusieurs classes, on utilise majoritairement la fonction d'activation **SoftMax** : $f(x) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$.


# Petit point sur les fonction de pertes

* Pour une régression linéaire, on dispose de plusieurs fonctions de perte comme **MSE** (*Mean Squared Error*) : $f(x) = \frac{1}{n} \sum_{i=1}^{n} (x_i - y_i)^2$ ou bien **MAE** (*Mean Absolute Error*) : $f(x) = \frac{1}{n} \sum_{i=1}^{n} |x_i - y_i|$.
* Pour une classification binaire, on dispose de fonctions de perte comme **Binary Cross-Entropy** : $f(x) = - \frac{1}{n} \sum_{i=1}^{n} (y_i \log(x_i) + (1 - y_i) \log(1 - x_i))$ 
* Pour une classification à plusieurs classes, on dispose de fonctions de perte comme **Categorical Cross-Entropy** : $f(x) = - \frac{1}{n} \sum_{i=1}^{n} y_{i} \log(x_{i})$ 

*x représente la valeur expérimentale attendue et y la valeur expérimentale prédite par le réseau de neurones.*


# Séquence 2 : Réseaux de neurones convolutifs (CNN)

## Principe
 
Un réseau de neurone convolutif se base sur l'utilisation de kernels (ou filtre). Ce sont des matrices de taille $kx \times ky$  multipliées tour à tour sur des parties de l'image initiale. Le produit présenté est un produit scalaire sur les matrices. image initiale est donc transformée en une nouvelle image de taille $n_x \times n_y$ avec $n_x = n_{x0} - k_x + 1$ et $n_y = n_{y0} - k_y + 1$.  

Lorsque l'image est en couleur, on utilise notamment plusieurs kernels (un par couleur).  

Finalement, chaque pixel de l'image résultat est passée dans une fonction d'activation pour n'obtenir qu'une seule valeur. Si on a plusieurs "couleurs", toutes les couches entrente en même temps dans la fonction d'activation pour ne donner qu'une seule valeur (bien qu'il y en ai plusieurs initialement).

La descente de gradient permet de trouver les meilleurs kernels pour notre image : les poids à optimiser sont donc les coefficients de la matrice du kernel + un biais global. Pour finir, la "forme" du kernel peut (en fonction des cas) faire ressortir des caractérisiques de l'image comme les contours des formes, les lignes droites, les contrastes  etc...


**Remarque :**  
* 1 neurone = 1 kernel + 1 biais ; le kernel est en 3 dimensions (2 dimensions pour la matrice + 1 dimension pour le nombre de plans convolutifs en entrée du neurone)
* En pratique on commence souvent par des kernel de plus grande dimension pour tendre vers de kernels de plus petite dimension.

## Paramètres des neuronens convolutifs

Il y a 3 paramètres pricipaux à définir pour un neurone convolutif :  

* La **taille du kernel** : La dimension de la matrice du kernel (donc lié à la)

* Le **padding** (bourage) : Comme notre image de départ est réduite en taille via la convolution, le padding permet de définir si on veut que notre image d'arrivée ai la même taille que celle de départ. Pour cela, on rajoutera des lignes et colones de 0 autour de l'image initiale (en l'agrandissant) pour qu'une fois la convolution réalisée, la taille reste la même.

* Le **stride** (le pas) : C'est le pas d'application de notre kernel. Si le pas augmentent, la taille de l'image d'arrivée diminue. Si le pas diminue, la taille de l'image d'arrivée augmente.  
Le stride se comporte de 2 entiers pour le pas selon $x$ et le pas selon $y$.

## Le réseau de neurone convolutif (CNN)

Chaque neurone agit de la façon suivante :
* Il reçoit n plans convolutifs issus de la couche de neurone précédente (pour la première couche, le neurone ne reçoit qu'une seule image : celle initiale).
* Pour chaque plan convolutif, il applique un kernel de taille $k_x \times k_y$ ce qui fait qui fait qu'il applique au total un kernel de taille $k_x \times k_y \times n$. L'application du kernel réalise le produit scalaire (en 3 dimensions) et le passage dans la fonction d'activation
* Une fois le kernel appliqué il se retrouve avec une image (1 seule alors qu'il y avait n plans en entrée) dont la taille dépend du padding et du stride.
* Sur cette image il réalise une opération de **pooling** (max pooling ou average pooling) qui consiste à réduire grandement sa taille. Pour faire cela, on dispose d'un facteur d'échelle qui va réduire la taille de l'image en fusionnant des pixel avec une fonction ($max$ ou $average$ par exemple).
* C'est cette image réduite qui sera ensuite injecté dans les neurones de la couche suivante.

A la fin, on possède plein de plans convolutifs de petite taille qui se sont "spécialisés" sur des caractéristique de l'image initiale !!

Finalement, on mets "à plat" toutes ces plans convolutifs et on les mets en entrée d'un réseau e neurone "classique" (DNN) pour réaliser la classification de l'image.

**Remarque :**  
Chaque neurone fait intervernir un nombre de paramètre égal à : $n*k_x*k_y + 1$. Soit le nombre de paramètre du kernel (en 3 dimensions) et le biais.
Il peut aussi être interessant de faire du **dropout** sur les neurones convolutifs pour éviter le sur-apprentissage. Cela consiste à "éteindre" aléatoirement certains neurones lors de l'apprentissage afin d'éviter que certains neurones soient trop important dans la classification finale.

# Séquence 3 : Réseaux de neurones convolutifs (partie 2) : l'importance de la préparation des données

## Motivations
* Lorsque l'on travaille avec des gros sets de données, il est souvent nécessaire de regarder ce qui se passe pendant l'apprentissage afin de s'assurer que le mùodèle est bon.
* Il est aussi important de regarder à l'avance ce à quoi ressemble nos données avec des arguments statistiques (moyenne, écart-type, médiane, etc...) ainsi que la représentation des différentes classes. Cela nous permettra entre autre de mieux appréhender les résultats intermédiaires lors de l'apprentissage.
* Il faut aussi regarder, notamment avec des images, ce à quoi ressemble nos données. En particulier, on peut avoir des soucis de tailles d'images : il faut que toutes aient le même format hauteur/largeur. On peut aussi avoir des soucis de couleurs : il faut que toutes les images soient en noir et blanc ou en couleur. On peut aussi avoir des soucis de résolution : il faut que toutes les images aient la même résolution. Il faut donc faire attention à ces points avant de commencer l'apprentissage. **Il est donc nécessaire de faire un pré-traitement des données avant de les utiliser pour l'apprentissage.**

## Préparation des données

Pour le cas de reconnaissances de panneaux, les réseaux de neurones CNN sont peu robuste pour reconnaître les panneaux tournés.

Pour des images, on peut notamment réaliser les traitements suivants :
* Augmenter l'exposition des images
* Réduire la résolution des images et les redimensionner au même format
* Mettre les images en noir et blanc
* Extrapoler les couleurs vers leurs extrêmes (pour les images en couleur)

Finalement, unne fois traitée, on peut enregistrer tous nos données dans un fichier au format `.h5`, ce qui permet d'accélerer le chargement des données. En plus de cela, pour réduire la taille du fichier, on peut les encoder en `float16` au lieu de `float32` (ou `float32` au lieu de `float64`) en retournant l'image numpy avec l'argument `dtype=np.float16` (ou `dtype=np.float32`).

## Suivie de l'apprentissage

Il s'agit de préparer des rappels ("callback" en anglais) afin de pouvoir conserver un historique de nos modèles et de sélectionner le plus performant. Ainsi on peut entre autre suivre l'évolution en temps réel de notre apprentissage (ce qui nous permet dès le début de voir si notre modèle est bon ou non car le temps de calcul peut être très long).

## Augmentation artificielle de données

Le dernier soucis que l'on peut avoir est la répartition de nos données d'un point de vue de l'effectif. Une classe pour laquelle on initiallement peu de données sera moinsprécise à la reconnaissance. Ainsi on peut augmenter le nombre de donnnées en modifiant légèrement ceux initiallement présents. Sur des images, cela peut se faire via des flous, des légères rotations etc...

# Séquence 4 : Les mathématiques derrière les réseaux de neurones
## Les kernels dans un réseau de neurone convolutif

Parmis les kernels classiques on peut trouver :
* $ \mathbf{K} = \begin{pmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{pmatrix} $ qui permet de détecter les contours verticaux (réalise une forme de dérivée ou de gradient sur l'image initiale)
* $ \mathbf{K} = \begin{pmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{pmatrix} $ qui permet de détecter les contours horizontaux (réalise une forme de dérivée ou de gradient sur l'image initiale)
* $ \mathbf{K} = \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix} $ qui permet de détecter les contours (réalise une forme de dérivée ou de gradient en deux dimensions sur l'image initiale)


## Importance du biais et des fonctions d'activations

Prenons l'exemple de la fonction ReLU définie : $f(x) = max(0, x)$ va grandement être influencée par le biais utilisé après notre convolution, notamment à cause de sa non-linéarité. Si le biais est tel que la valeur est tout le temps négative, alors l'image sera unie (noire) et ne possèdera que des pixels égals à 0. On perd donc toute l'information de l'image. À l'inverse, un biais suffisament élevé, permettra de bien conserver l'information. Un entre-deux permettra de ne conserver que certaines valeurs de gradient / dérivée / valeur max de chaque image uniquement où cela est nécessaire. ReLU va en quelque sort réaliser un passe-haut sur notre image pour ne conserver que les valeurs importantes.  

La fonction Softmax : $f(x) = \frac{e^x}{\sum_{i=1}^n e^{x_i}}$ va permettre de normaliser les valeurs de sortie de notre réseau de neurones et le tranformer en une distribution de probabilité. Ainsi, on pourra plus facilement comparer les différentes valeurs de sortie entre elles. Cela permet aussi de mieux visualiser les résultats de notre réseau de neurones.

## La descente de gradient

Derrière cette notion se cache la question de coût ou de biais de notre réseau de neurone. Lorsque l'on possède plusieurs neurones $j$ (ayant chacun les paramètres $w_i^{j}$ et $b^{j}$), le coût de notre réseau sera une fonction de tout ces paramètres : $cout = f(w_1^1, w_2^1,...,w_N^1,b^1,w_1^2,...,w_N^M,b^M)$. Pour réaliser ce calcul, on est ammené à regarder le gradient de la fonction coût. Or, dans cette cette fonction, on retrouve une composition de fonction liée à l'architecture de notre réseau de neurones : composition linéaire et fonction d'activation. On est alors amené à utiliser la règle de la chaîne pour connaître la dérivée du coût selon un paramètre $w_i^j$ : $\frac{\partial cout}{\partial w_i^j} = \frac{\partial cout}{\partial z^j} \frac{\partial z^j}{\partial w_i^j}$. $z^j$ étant la sortie du neurone $j$ avant la fonction d'activation, on connaît donc $\frac{\partial z^j}{\partial w_i^j}$ qui est liée à la structure du  eurone $j$. Pour trouver $\frac{\partial cout}{\partial z^j}$, on va réutiliser la règle de la chaine, et ainsi de suite. C'est l'algorithme de **rétropropagation de l'erreur**.

Dans l'algorithme de rétropopagation de l'erreur, on va calculer nos dérivées partielles en partant de la sortie de notre réseau. Cela est un calcul rapide pour le gpu, mais qui peut être ralenti par la taille de nos données.

## La descente de gradient stochastique

Lorsque l'on possède un grand nombre de données, il est très coûteux de calculer le gradient de la fonction coût pour toutes les données. On va alors utiliser un sous-ensemble de nos données pour calculer le gradient. Ainsi, en calculant le gradient pour chaque sous-ensemble de données et en mettant à jour nos paramètre et ce pour chaque génération jusqu'à ce que l'on ait parcouru toutes nos données, on va obtenir notre descente de gradient plus rapidement. On appelle cela une **époque**. On peut alors répéter cette opération plusieurs fois pour améliorer notre modèle : c'est la **descente de gradient stochastique**.

# Séquence 5a : Evalutation des modèles

*Note : Réaliser des optimiseurs à un ordre plus élevé que l'ordre 1 pour remplacer nos neurones est encore à l'étape de la recherche mathématique.*

## Principe d'évaluation d'un modèle

Lors de notre conception de modèle, on utilise un certain nombre d'hyperparamètres sur lesquels on peut jouer afin de modifier les performances de notre réseau de neurone (nombre de couche, nombre de neurone, taille de kernel, fonction d'activation etc ...). cependant, tester tputes les différentes impossible est long. À chaque fois, on prends environ 70% du data set pour l'entrainement, 15% pour la validation de l'apprentissage, et 15% pour faire une évaluation finale de notre modèle afin d'obtenir sa performance. De plus, à chaque fois que l'on fait l'apprentissage, il faut bien penser à mélanger notre dataset afin de ne pas avoir des données trop similaires à la suite. Finalement, on obtient une performance en forme de gaussiène pour la modification d'un paramètre.  

Pour obtenir de meilleurs performances, on peut réaliser une **validation croisée**. On va séparer notre dataset en $k$ sous-ensembles. Puis, on va réaliser $k$ apprentissages en utilisant à chaque fois un sous-ensemble différent pour la validation. On obtient alors $k$ performances différentes pour lesquelles on peut calculer la moyenne et l'écart-type. On obtient alors une performance plus stable et plus représentative de notre modèle. 

On peut combiner cette méthode à la vision itérative avec brassage des données pour obtenir une meilleure performance.

Finalement, on peut regarder la performance de notre modèle avec différents indicateurs :
* La matrice de corrélation qui affiche les vrais positifs, les faux positifs, les vrais négatifs et les faux négatifs
* F1-score : Qui permet aussi de bien appréhender ces notions de faux-positifs etc ...
* La précision
* L'exactitude

# Séquence 5b : Les données creuses

Dans cette partie on va traiter le cas de données textuelles. Une des façpns de faire est de créer un dictionnaire qui pour chaque mot va encoder une une valeur entière. Ainsi, une phrase se traduit par un vecteur d'entier. Cependant, cette représentation est inutilisable, car notre vecteur d'entier ne possède aucun sens.  

Pour éviter se problème, on va créer un vecteur creux. Pour chaque mot, on aura un vecteur rempli de 0 sauf un 1 en l'indice de du mot qui correspond. Ainsi, 1 mot = 1 vecteur de dimension N (= la taille du dictionnaire) avec que des 0 sauf un 1 en l'indice du mot. Le soucis est que l'on a maintenant une structure de données  de très grande dimension et très creuse (beaucoup de 0). Alors, un mot est représenté par une dimension de N.

Pour traiter ces données on peut :
* Transformer la matrice en un vecteur avec l'application logique **ou** : cela a pour effet de trduire notre en phrase en un vecteur qui ressence uniquement la présence du mot dans la phrase indépendament de sa position. Ce vecteur est alors de la même dimension que le dictionnaire. Mais cette application reste cependant très limitée.
* Transformer un mot (vecteur creux) avec un vecteur de petite taille dont la valeur donne un vrai sens au mot qu'il représente. C'est l'**embedding** Ainsi, le sens des mots doit être transcrit dans les valeurs du vecteur dense construit (ex : Paris est à la France ce que Dakar est au Sénégal). Keras le fait très bien. L'embedding vas viser et se spécialiser dans un type de mot / phrase. Si on entraine notre modèle à étudier des recettes de cuisine, il sera inefficace pour étudier lestextes de droit.

## Construction d'un embedding
 
Pour construire un embedding, on dispose de 2 méthodes :
* **Continuous Bag of Words (CBW) :** Il s'agit de construire un mot en fonction de ses voisins. On prend pour chaque mot un ensemble de mot voisin de lui (souvent les 2 précédents et les 2 suivants) et on va entrainer notre réseau à trouver le mot au milieu en fonction de ses voisins.

* **Skip-Gram :** On part d'un mot et on entraine notre modèle à trouver les mots qui sont habituellement proches de lui. C'est d'une certaine façon l'approche opposée au CBW.

# Sequence 6 : Les réseaux de neurones récurrents (RNN)

## Principe
L'objectif d'un RNN est de traiter des séries de données (ex : texte, son, vidéo, etc ...). Pour cela, on va utiliser un neurone qui va se réinjecter sa sortie en entrée. Cela permet de traiter des données évolutives (notammment en fonction du temps pour les séries temporelles).

Un neurone classique se présente de la façon suivante :
$y = \sigma(W_x \cdot X + b)$ avec $W_x$ et $b$ le vecteur des poids et le biais.  

Un neurone récurent part du principe que la sortie du neurone sera réinjectée en entrée de celui-ci avec un autre jeu de poids : $y_t = \sigma(W_x \cdot X_{(t)} + w_y \cdot y_{(t-1)} + b)$ avec $w_y$ le vecteur des poids (scalaire car $y$ est ici scalaire) de la récurrence.

Dans les faits, on ne va pas considérer qu'un seul neurone, mais une cellule contenant plusieurs unités. Dans chaque cellule on dispose d'une entrée vectorielle et d'une sortie vectorielle (composée de la sortie scalaire de chaque unité). La sortie de chaque cellule se fait de lafaçon suivante : $Y_(t) = \phi(W_x \cdot X_{(t)} + W_y \cdot Y_{(t-1)} + B)$ avec $W_x$ et $W_y$ des matrices, $X$, $Y$ et $B$ des vecteurs et $\phi$ une fonction d'activation. On a ici augemnté la dimension de chaque objet car on prend en compte plusieurs unités dans une même cellule. À chaque itération, chaque unité de la cellule va générer une sortie scalaire composante du vecteur $Y$ de sortie.

Pour fonctionner correctement, un réseau récurent nécessite en entrée une série de vecteur d'entrée : $X_{(t_0)}$, $X_{(t_1)}$, $X_{(t_2)}$, ... $X_{(t_n)}$ ; et fournie en sortie : $Y_{(t_0)}$, $Y_{(t_1)}$, $Y_{(t_2)}$, ... $Y_{(t_n)}$.

$W_x$ dépend de la taille du vecteur $X$ en entrée et du nombre d'unité utilisées dans la cellule. $W_y$ dépend du nombre d'unité utilisées dans la cellule et de la taille du vecteur $Y$ en sortie (égal aunombre d'unité car une unité donne une coposante du vecteur $Y$). $B$ dépend du nombre d'unité utilisées dans la cellule. Ainsi :
* $dim(W_x) = dim(X) \times nb_{unit}$
* $dim(W_y) = nb_{unit} \times nb_{unit}$
* $dim(B) = 1 \times nb_{unit}$
* $dim(Y) = 1 \times nb_{unit}$

## Problème
On a ici la description d'un réseau récurrent d'ordre 1, c'est à dire qu'on ne considère que la valeur précédente pour la valeur suivante. Cela ne marche donc pas. De plus, réaliser un réseau récurrent "fort" qui considérerait toutes les valeurs précédentes est impossible car cela représente beaucoup trop de données.

## Long Short-Term Memory (LSTM)
La solution est de garder en mémoire deux choses :
* La valeur précédente de la sortie (court terme)
* Une valeur de mémoire construite à partir de toute les valeurs précédentes (long terme)

Ainsi, on a à chaque étape 3 entrées :
* La valeur d'entrée
* La valeur de sortie précédente (court terme)
* La valeur de mémoire (long terme)
Et on a aussi 2 sorties :
* La valeur de sortie (réutilisée pour le court terme)
* La nouvelle valeur de mémoire (utilisée pour le long terme)

**Il existe aussi les GRU (Gated Recurrent Unit) qui sont une version simplifiée et plus récente des LSTM.**

## Différent type d'utilisation :
* **Serie to Serie** : On a une série de vecteur en entrée et on veut une série de vecteur en sortie (on possède $X_0$ à $X_n$ et on veut $Y_0$ et $Y_n$)
* **Serie to Vector** : On a une série de vecteur en entrée et on veut un vecteur final en sortie (on possède $X_0$ à $X_n$ et on veut $Y_n$)
* **Vector to Serie** : On a un vecteur en entrée et on veut une série de vecteur en sortie générée à partir du seul vecteur d'entrée (on possède $X_0$ et on veut $Y_0$ à $Y_n$)
* **Encoder-Decoder** : On a une série de vecteur en entrée et on veut une série de vecteur en sortie qui seront générés à la suite de la série d'entrée (on possède $X_0$ et $X_1$ et on veut $Y_2$ et $Y_3$)

## Entrainement d'un RNN
On prend notre suite séquentielle de données et on la scinde en deux parties :
* La première partie (causale) deviendra notre donnée d'entrainement
* La seconde partie (conséquence) deviendra notre donnée de test (validation)

Ensuite, on va découper nos deux sets en plus petites séquences et les mélanger entre elles. L'entrainement visera à, pour chaque séquence, en fonction du début de celle-ci de déterminer la suite de la séquence. Ainsi, on va entrainer notre réseau à prédire la suite de la séquence à partir de son début.

**Attention :** Le passé peut-il expliquer le future ? Si oui, un tel réseau peut faire sens, sinon, ça ne fonctionnera pas.

# Sequence 7 : PyTorch
PyTorch se trouve, en terme de difficulté, à mi-chemin entre Tensorflow et Keras. Il est plus bas-niveau que Keras, mais moins que Tensorflow.

## PyTorch tensors
Tout le travail avec PyTorch se situe autour de l'objet `torch.Tensor`. C'est un objet qui ressemble beaucoup à un `numpy.ndarray` mais qui possède des propriétés supplémentaires.

```python
import torch
import numpy as np
t1 = torch.tensor([[1, 2, 3],[4, 5, 6]])
t2 = torch.tensor(np.array([[1, 2, 3],[4, 5, 6]]))
t3 = torch.zeros((2, 3), dtype=torch.float32) # On peut naturellement définir le type de données
```
**Attention :**
```python	
t1[0, 0] # retourne "tensor(1)"
t1[0, 0].item() # retourne "1"
```
De façon générale, il y a pas mal de diufférence de syntaxe mais rien de bien compliqué. Le gros aventage de PyTorch est qu'il permet de faire de la différenciation automatique lors de l'entrainement de réseau de neurones. C'est à de calculer le gradient de la fonction de perte par rapport aux paramètres du réseau.

 Une autre différence est de pouvoir basculer facilement entre CPU et GPU : Pour des gros calculs de prédiction ou d'entrainement, on peut basculer manuellement sur le GPU pour les réaliser. Il faut bien penser à revenir sur le CPU pour afficher les résultats.