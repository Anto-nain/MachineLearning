


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

# Séquence 6 : Les réseaux de neurones récurrents (RNN)

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

# Séquence 7 : PyTorch
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

# Séquence 8 : Les transformers
Les transformers sont utilisés majoritairement pour le traitement du langage naturel, mais aussi dans les autres domaines (traitement d'image etc...). Avec les anciennes méthodes d'embedding, on pouvait réaliser des opérations sur des mots : $woman + (king - man) \approx queen$.
## Principe
Pour donner toutes les informations necessaires au tranformer, on va lui fournir : 
* Une information sur les mots : **embedding**
* Une information sur la position des mots dans la phrase : **positional encoding**. Pour cet e,codage, un des premières fonction arrivée (en 2017) a été d'utiliser des fonctions $sinus$ et $cosinus$. Derrière cela se cache le fait qu'en binaire le premier bit est inversé (de 0 à 1 ou inversement) à chaque incrément, le second tout les 2 incréments, puis tout les 4 etc... en suivant une fonction périodique. D'où l'utilisation de fonctions trigonométriques pour encoder la position des mots. En gros, cela permet de compter en binaire dans un espace continue.

## Historique
* En premier on utilisait des CNN pour faire une convolution 1D. Mais très vite, on s'est rendu compte que cela ne fonctionnait pas très bien car on est limité par la taille de fenêtre pour la convolution et dès lors qu'on a des phrases assez longues, on perd beaucoup d'information.
* En second, on a utilisé des réseaux récurrent(RNN) qui travaillaient en fonction des mots précédents mais aussi des mots suivants : de façon bi-directionnels avec un RNN dans un sens et un autre dans l'autre sens. Ainsi, on avait un réseau encoder qui travaillait pour comprendre la phrase d'entrée et un décoder qui générait par RNN la phrase de sortie en se bassant sur le réseau d'entrée.

## Structure NLP (Natural Language Processing)
La structure d'un système NLP (Natural Language Processing) est la suivante :
* **Tokenization** : On découpe la phrase en mots
* **Pre-processing (Embedding)** : On encode les mots que ça soit le token, mais aussi la position de ce dernier dans la phrase
* **Transformer model (Encoder)** : On passe la phrase dans le transformer pour obtenir une série de vecteur où l'information de la phrase y ets bien dispatchée
* **Task Head (Decoder)** : On passe la série de vecteur dans le décodeur (*Task Head*) pour réaliser la tâche souhaitée
Qu'y a-t-il dans le transformer ?

En entrée du transformer il y a une séquence de vecteur et en sortie il y a aussi une séquence de vecteur.

Dans le transformer on retrouve plusieurs couches (dont on a montré par l'expérience que la modification des hyperparamètres d'une couche à l'autre ne changeait pas grand chose)

Dans chaque couche on retrouve :
* **Multi-Head Attention** : 
* **Add & Norm** : Qui va ajouter et normaliser les données de la couche e sortie avec la couche d'entrée de la couche multi-head attention (pour assurer la bonne convergence de notre modèle)
* **Feed Forward** : Qui est une petite couche de réseau dense (1 à 2 couches)
* **Add & Norm** : Comme précédement en ajoutant l'entrée de la couche de feed-forward.

## Self Attention
En entrée on a une série de vecteur de même taille.

Chaque vecteur d'entrée va passer dans 3 réseau de neurones dense en parallèle. Ce qui va créer 3 matrices (une dimension pour la taille des vecteurs d'entrée souvent égale à celle de sortie, une dimension pour le nombre de vecteur d'entrée toujours égale au nombre de veteur de sortie). Ces 3 matrices sont :
* **Query (Q)**
* **Key (K)** dont toutes les valeurs sont multipliées par un facteur $\frac{1}{\sqrt{d_k}}$ où $d_k$ est la longueur des vecteurs de sortie pour chaque mot (un des dimension de la matrice)
* **Value (V)**

Ensuite, on va calculer le produit entre $Q$ et $K^T$ pour obtenir la matrice des produits scalaires entre les mots. Ensuite on va passer cette matrice dans un fonction *SoftMax*  et obtenir la matrice d'attention : $AttentionMatrix = SoftMax(Q \times K^T)$. Cette matrice permet de mettre en avant les mots proches et impportants dans la phrase.

Afin d'obtenir la matrice de sortie, on va multiplier la matrice d'attention avec la matrice $V$ et ainsi obtenir notre séquence de sortie. Ainsi, chaque vecteur de la séquence de sortie est une combinaison de l'information contenue dans la séquence d'entrée : $OutputMatrix = AttentionMatrix \times V$. Cela permet de transformer nos mots importants en vecteur de sortie importants.

## Multi-Head Attention
Lors du passage de la séquence d'entrée dans les réseaux denses pour créer nos matrices $Q$, $K$ et $V$, l'objectif va être de les séparer plusieurs entités (têtes) selon la dimension de la taille des vecteurs d'entrée (et non pas selon les nombre de vecteurs dans la séquence de sortie) En suite, maintenant que l'on a nos matrices ${Q_1, Q_2, ..., Q_n}$, ${K_1, K_2, ..., K_n}$ et ${V_1, V_2, ..., V_n}$ on peut réaliser les mêmes opération que précédement avec nos $n$ têtes et obtenir $n$ sorties : ${O_1, O_2, ..., O_n}$ qui une fois concaténées donnent la sortie de notre couche de multi-head attention. L'importance de ces multiples têtes est de ne pas oublier de l'information lors de la transformation de la séquence d'entrée en séquence de sortie car avec une seule tête on ne garde que l'information la plus importante ce qui induit beaucoup de perte.

## Modèle de type auto-encoder / encodeur-décodeur
Dans le principe vu plus haut, les matrices $Q$, $K$ et $V$ dans nos transformer ne travaillent qu'entre elle. ainsi, il est difficile de faire passer cette information au tranformer qui va décoder la sortie d'où l'importance de parfois d'ajouter l'information des matrices $Q$,$K$ et $V$ de l'encoder dans celles du décoder.

## Entrainement
Il est important de pré-entrainer notre modèle pour qu'il soit capable de correctement reconnaitre le langage afin d'éviter de repartir de 0 pour chaque nouvelle tâche. Dans l'exemple de Bert, le modèle de transformer a simplement été entrainé à reconnaitre des mots cachés. Pour lui faire faire des tâches plus complexes on va travailler sur la task head.

## Comment choisir la tâche à réaliser ?
Notre tranformer a déjà été entrainé. Il ne nos reste plus qu'à le faire travailler sur la couche Task Head pour lui faire faire ce que l'on veut. Par exemple, pour de la classification de texte, on va définir un petit réseau de neurone qui passera dans un softmax pour obtenir une probabilité de classe. On a ici un second réseau qui va se spécialiser sur la tâche à réaliser. Ainsi, le transformer va simplement générer du texte et lui donner donner du sens à partir de celui en entrée, et c'est la couche task head qui va s'entrainer à bien reconnaitre telle ou telle classe.

De plus, en amont du texte d'entrée, on peut forcer le rajout d'un template de texte pour que notre modèle se spécialise sur certaines actions.

## Visual tranformers
Utiliser la structure de transformers, où l'on fait prédire une partie d'image à partir d'une autre, fonctionne ârfaitement avec la structure de transformers. 

# Séquence 9 : Graph Neural Networks (GNN)
Dans les images ou dans les textes, les données sont très strucutrées : chaque pixel a 8 voisins (moins pour les bords ou les coins), chaque mot est précédé et suivi d'un autre mot. Cependant, il existe des formes de données plus complexes comme les molécules, les maillages ou autre. Pour ces données on parle de géométric Deep Learning. Une grande partie de ces données peuvent être représentés dans des graphes.

## La complexité
Il est très facile d'avoir des graphes rapidement complexes avec énormement de noeuds et d'arrêtes. Finalement, il est important de comprendre sur des structures géométriques simples en sachant que cela peut être étendu à des structures plus complexes.

## Les graphes
Les graphes sont composées de :
* **Noeuds** : Qui sont des entités qui peuvent être de différentes nature (molécules, personnes, etc...)
* **Arrêtes** : Qui sont des liens entre les noeuds et qui peuvent être de différentes nature (type de liaison chimique, relation entre personnes, etc...)

Ils peuvent être de diffrente nature :
* **Non orientés** : La relation entre deux noeuds est réciproque dans les deux sens : A <-> B
* **Orientés** : La relation entre deux noeuds est différente d'un sens à l'autre : A -> B

Ainsi, la notion de chemin d'un noeud à l'autre n'a pas la même complexité que l'on soit dans un graphe orienté ou non. De plus, on peut parler de cycles dans un graph orienté lorsqu'il eiste un chemin permettant de relier un noeud à lui même.

Finalement, on va pouvoir stocker de l'information dans les noeuds, les arrêtes et les graphes tout entier. Par exemple, dans le cas de molécules, on peut stocker l'information des atomes dans les noeuds, l'information des liaisons chimiques dans les arrêtes et l'information de la molécule dans le graphe.

## Mesurer la structure d'un graphe
Il existe plusieurs manières de mesurer la structure d'un graphe :
* **Node Proximity** : Qui mesure la proximité entre deux noeuds. Cela peut se faire de différentes façons comme le nombre de noeud qui séparent les deux noeuds, ou bien la valeur du plus petit chemin entre les deux noeuds.
* **Node centrality** : Qui mesure le nombre de chemin passant par un noeud. Cela permet de mesurer l'importance d'un noeud dans le graphe.

## Représenter un graphe
Il existe plusieurs manières de représenter un graphe :
* **La matrice d'adjacence** : Qui est une matrice carrée de taille $n \times n$ où $n$ est le nombre de noeuds du graphe. Ainsi, chaque ligne et chaque colonne représente un noeud et la valeur de la case $(i,j)$ représente la valeur de l'arrête entre le noeud $i$ et le noeud $j$. Cette matrice est symétrique dans le cas d'un graphe non orienté.
* **La liste d'adajacence** : Qui est une liste stockant les paires de noeuds reliés par une arrête.

**Problèmes** : La matrice d'adjacence est de taille $nb_{noeuds}^2$ ce qui peut causer des problèmes de stockage, sachant qu'elle est assez creuse. La complexité pour trouver les arrêtes dabs une liste d'adjacence est bien plus élevée que pour la matrice d'adjacence.

## Matrices usuelles
* **Adjacence (W)** : La matrice d'adjacence du graphe stockant le poids des arrêtes.
* **Degré (D)** : La matrice diagonale stockant sur la diagonale le degré de chaque noeud, c'est à dire le nombre d'arrêtes reliées à ce noeud.
* **Laplacienne (L)** : La matrice $L = D - W$.
* **labels (X)** : La matrice stockant les labels des noeuds.

## Comment travailler sur des graphes ?
Il s'agit de réaliser de l'embedding de graphe.
* Dans un premier temps, à l'aide de nos matrices $X$ et $W$, on va transposer à l'aide d'un embedding nos informations dans un espace latent $Z$.
* Ensuite, on va décoder ces informations pour prédire $\hat{y}$ qui va regrouper les information que l'on cherche à prédire sur notre graphe (une valeur, une classe, etc...).
* On va aussi décoder notre graphe pour essayer de reconstruire notre matrice d'adjacence $\hat{W}$ (chercher une sorte de réciproque à l'embedding).

Toute la difficulté est de trouver la bonne dimension de l'espace latent $Z$ (qui est un hyperparamètre de notre modèle). En cas d'un dimension trop petite, on ne pourra pas bien séparer les informations de notre graphe en fonction de la tâche à réaliser (sous-apprentissage). En cas d'une dimension trop grande, on va avoir un sur-apprentissage et notre modèle va apprendre des informations inutiles.

## Entraînement inductif // transductif
L'entraînement inductif consiste à entraîner notre modèle sur plein de graphes, puis avec notre modèle entrainé on va prendre un graphe et lui appliquer la tâche pour laquelle notre modèle a été entraîné.

Pour un graphe il peut être intéressant de réaliser un entraînement transductif. Cela consiste à entraîner notre modèle sur un seul graphe. On va entraîner notre modèle sur des portions connues de ce graphe, puis, on va utiliser notre modèle pour prédire des portions manquantes de notre graphe. Cela permet notamment de faire du Node Labeling, c'est à dire de complèter la valeur de noeuds dont on ne connnait pas la valeur grâce au reste du graphe. Il faut faire attention, car ce modèle est alors spécialisé sur un seul graphe et ne peut pas être utilisé sur d'autres graphes.

## Tâches possibles
* **Clustering (mettre des labels sur les noeuds)** : Qui consiste à regrouper les noeuds en fonction de leurs similarités (ex : regrouper les personnes en fonction de leurs relations ou détecter des bots).
* **Link making (mettre des arrêtes entre les noeuds)** : Qui consiste à prédire les relations entre les noeuds. Utilisé par Alphafold pour préduire le recouvrement 3D des protéines, (en reliant les molécules entre elles).
* **Graph classification** : Qui consiste à prédire une valeur sur le graphe entier (ex : prédire le type de molécule, sa dangerosité, etc...).

## Différents types de réseaux sur les graphes
* **Convolution de graphe** : Puisque le nombre de voisin n'est pas fixe, la création de filtres est plus compliquée. Il s'agit donc d'utiliser des opérateurs invariant par permutation (ex : moyenne, max, etc...). Le point fort est que l'on peut facilement récupérer les informations de noeuds voisins éloignés, il peut donc être intéressant de mettre en place un *cut-off* pour ne pas prendre en compte les noeuds trop éloignés. On peut même créer un *virtual node* qui est relié a tout les noeuds du graphe pour permettre de passer des informations entre les noeuds éloignés.
* **Message Passing** : L'objectif est de créer une tranformation que le réseau va pouvoir apprendre qui permettra de transformer après embedding, un vecteur de l'espace des arrête en vecteur de l'espace des noeuds ou en vecteur de l'espace de graphes. Ainsi, par exemple, on va pouvoir utiliser les informations sur les noeuds pour mettre à jour les informations sur les arrêtes, alors que cela était impossible.
* **Graph Tranformer Network** : Il est aussi possible d'utiliser des transformers sur les graphes. La structure est la même qu'un tranformer classqiue mais on doit bien s'assurer de pouvoir comparer les informations des noeuds et les informations des arrêtes.

# Séquence 10 : Les réseaux auto-encoders (AE)
## Principe
L'objectif est de partir d'une information en entrée qui va être réduite en toute petite dimension dans un espace latent (encodage), puis qui va être retransformée dans l'espace recherché (décodage). Une bonne façon de visualiser cela est de s'imagine une image bruitée en entrée que l'on va essayer de débruiter.

On dispose alors de deux parties dans notre réseau :
* **L'encodeur** : Qui va réduire la dimension de notre information jusqu'à la taille de notre espace latent (typiquement utilisé pour des images donc on travail notamment avec un réseau convolutif, mais pas forcément)
* **Le décodeur** : Qui va transformer la représentation de notre entrée dans l'espace latent en une représentation dans l'espace de sortie.

Ici on parle d'apprentissage auto-supervisé. *Supervisé* car on possède initialement une image bruitée et une image non bruitée. *Auto* car on a pas besoin qu'un humain vienne nous dire si notre image est bruitée ou non.

## Convolution transposée
Lorsque l'on se trouve dans le cadre d'un réseau convolutif, on peut réaliser la transposée d'un convolution (qui fonctionne de façon symétrique à la convolution). C'est à dire qu'au lieu d'utiliser les kernels pour réduire la dimension de notre image, on va l'augmenter. 

Le sens des paramètres de **stride** et de **padding** ne sont plus du tout les mêmes. Ainsi, pour une convolution, un stride de 2 va réduire la taille de notre image par 2, alors que pour une convolution transposée, un stride de 2 va augmenter la taille de notre image par 2. De même, pour une convolution, un padding de 1 va ajouter un pixel de bordure à notre image, alors que pour une convolution transposée, un padding de 1 va enlever un pixel de bordure à notre image.

Une autre façon d'augmenter la dimension de notre image est d'utiliser des couches de **UpSampling** qui vont simplement répéter les pixels de notre image pour augmenter sa taille, par exemple :
$ \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \rightarrow \begin{bmatrix} 1 & 1 & 2 & 2 \\ 1 & 1 & 2 & 2 \\ 3 & 3 & 4 & 4 \\ 3 & 3 & 4 & 4 \end{bmatrix}$

## Multiple input/output model
Il est possible de créer des modèles à plusieurs entrées. Cela peut être utile pour créer des modèles qui prennent en entrée des données de nature différentes (ex : image + texte). Pour faire cela, il suffit simplement de concaténer les entrées une fois qu'elles ont été traitées par leur encodeur respectif.

Deplus, cela permet de créer des modèles avec de multiples sorties pour des tâches différentes. Par exemple, pour un modèle qui prend en entrée une image et qui doit prédire la classe de l'image, mais aussi la débruiter.


## Inception model

Il s'agit d'avior plusieurs chemins internes qui vont chacun traiter l'information de façon différente. À la fin, il faut utiliser une conche de concaténation que va rassembler toutes nos informations pour les traiter ensemble.



# Séquence 11 : Les Variational Auto-Encoders (VAE)
Retour sur le prince d'un AE : On part d'une information en entrée qui va être encodé et réduite dans un espace latent "Z" puis décodé pour revenir dans l'espace de sortie. Avec ```keras```, la structure est très simple :
```python
z = encoder(inputs)
outputs = decoder(z)
```
On avait notamment noté que la vision de l'espace latent permettait de bien visualiser la distribution de nos donnnées. Dans le cas d'une classification, en projettant nos données de l'espace latent on visualisait bien les clusters des différentes classes.

## Principe d'un VAE 

Le principe est de réussir à obtenir pour chaque dimension de de l'espace latent :
* Une moyenne $\mu$ 
* Un écart-type $\sigma$
Ainsi on possède deux vecteurs de la même dimension que l'espace latent qui représentent la distribution probabiliste de nos entrées dans l'espace latent.

Une fois que l'on possède ces vecteurs, on va piocher alléatoirement un point, suivant la distribution que l'on vient de réaliser, et vérifier si il sera décodé  comme il le faut.

Finalement, notre réseau va apprendre à générer $\mu$ et $\sigma$ pour que losrque l'on ensuite un point avec ces paramètres, il soit correctement décodé.

## Calcul de la perte de notre réseau

Dans une grande partie des cas, on souhaite reconstruire notre entrée initiale avec la sortie de notre VAE. Il s'agit donc de calculer la perte par reconstruction de cette dernière $loss_{reconstruction}$ ; très souvent via une *cross-entropy*.

Parallèlement à cela, on peut aussi se retrouver avec un éparpillement dans l'espace latent des distributions générées par $\mu$ et $\sigma$, chose que l'on souhaite limiter. On va donc aussi prendre en compte un terme de régularisation qui va limiter cet éparpillement. Il s'agit de la divergence de Kullback-Leibler (KL) qui permet de mesurer la distance entre deux distributions. Une expressions simplifiée de cette divergence est :
$D_{KL} = \frac{1}{2} \sum_{i=1}^{n} (\sigma_i^2 + \mu_i^2 - 1 - log(\sigma_i^2))$ avec $k$ la dimension de l'espace latent (et donc des vecteurs $\mu$ et $\sigma$).

Finalement, on va construire notre fonction de perte en appliquant deux facteurs pour chosir l'importance de chaque perte :

$loss = \alpha_{rec}loss_{reconstruction} + \alpha_{KL}D_{KL}$

Enfin, il faut faire attention au coefficient que l'on donne pour chaque perte :
* Si on prend trop en compte la reconstruction, cette dernière sera efficace, et on verra bien apparaitre les clusters dans l'espace latent : les distributions ne se recouvrent que très peu (d'où la bonne reconstruction). Cependant, les valeurs de $\mu$ et $\sigma$ seront très éparces : des clusters seront ultra denses et centrés ; d'autres seront extrêmement éparpillés et centrés loin de l'origine.
* Si on prend trop en compte la régularisation, toutes nos distributions seront bien centrées en 0 avec un écart-type proche de 1, mais elles se chevaucheront alors toutes et on ne pourra plus distinguer les clusters dans l'espace latent. De ce fait, la reconstruction sera très mauvaise.

De bons coefficients s'assurent que les clusters se font proche de 0 avec des écart-types proches de 1, tout en restant bien séparés les uns des autres sans trop se chevaucher.

## Comment générer de nouvelles données ?
Maintenant que l'on possède des classes nos trop éparpillées, non trop chevauchantes, on peut à l'aide d'une grande quantitée de points, générer un grande quantité de donnée sur notre espace latent une fois décodée. Et ce avec une grande précision.

# Séquence 12 XXX

# Séquence 13 : Les Generative Adversarial Networks (GAN)
## Principe
Le principe est de faire s'affronter deux réseaux de neurones :
* L'un va produire des donnnées
* L'autre va s'entrainer à visualiser si les données sont réelles ou non

## Entraînement
On va avoir un processus d'entraînement en deux temps. Il faut entraîner successivement le générateur, puis le discriminateur, puis le générateur, puis le discriminateur, etc... jusqu'à ce que l'on obtienne un résultat satisfaisant.
* Du point de vue du discriminateur, il s'agit de recevoir des données en entrée et de dire si elles sont vraies ou fausses (classification binaire).
* Du point de vue du générateur, on labelise nos données comme étant vraies, puis si elles sont annoncées comme vraies (victoire) on a bien dupé de discriminateur, sinon (défaite) on a perdu et on définit notre perte comme la victoir du discriminateur. À noter que nos données d'entrée sont générées aléatoirement.

**Attention** : Le GAN sont très gloutons en terme de ressources. Il faut donc bien penser à les utiliser sur des données de petite taille.

Pour faire la rétropropagation dans keras, il peut être intéressant de faire le calcul du gradient à la main avec ```Tensorflow```

## Amélioration des GANs
Les GAN standards sont assez fragiles notamment à caude de la disparition du gradient, ce qui va donner lieu à certaines améliorations :
* Un nouveau calcul du coût : Earth Mover Distance (EMD) ou Wasserstein qui va calculer la distance entre les distributions de probabilité des vraies et fausses données. Le calcul est très compliqué et lourd à mettre en place. Cela donne un réécriture du coût :

  $loss_{critic} = \frac{1}{m}\sum_{i=1}^{m}[D(x_i) - D(G(z_i))]$

  $loss_{generator} = \frac{1}{m}\sum_{i=1}^{m}[D(G(z_i))]$

  avec $m$ le nombre de données en entrée, $x_i$ les données réelles, $z_i$ les données générées et $D(x)$ la sortie du discriminateur pour des vraies images, G(z) la sortie du générateur pour une valeur z de l'espace latent tel que G(z) ressemble à une vraie image alors que c'est une fausse et D(G(z)) la sortie du discriminateur pour une fausse image.

  Il s'agira ici de faire en sorte que la critique pour une vraie image soit bien meilleure que pour une fausse image tout en concervant l'idée que le générateur arrive bien à tromper le discriminateur. De plus, en limitant la valeur des poids dans [-0.01, 0.01] (**on parle de clipping**), on s'assure la caractéristique k-Lipschitzen du discriminateur, assurant ainsi sa caractéristique bornée.

* L'ajout de pénalité du gradient. Un terme supplémentaire vient se rajouter dans la fonction de coût du discriminateur :

  $loss_{critic} = \frac{1}{m}\sum_{i=1}^{m}[D(x_i) - D(G(z_i))] + \lambda \frac{1}{m}\sum_{i=1}^{m}[(||\nabla D(\hat{x_i})||_2 - 1)^2]$

  avec $\hat{x_i} = \alpha x_i + (1 - \alpha)G(z_i)$, $\alpha$ un nombre aléatoire entre 0 et 1 et $\lambda$ le coefficient de la pénalité du gradient. Cela permet de s'assurer de la bonne convergence du gradient sans avoir à faire du clipping.

# Séquence 14 : Les Diffusion Probabilistic Models (DPM)
Il s'agit de modèles ayant permis les dernières avancées dans le domaine de la génération d'images à partir de textes.

Avec les GANs et les VAEs, on a vu de type de modèles génératifs. De plus, on peut les évaluer de deux façons : 
* $log(P(x))$ ou en français "log-vraissemblance" : La probabilité que notre modèle génère une image $x$.
* $FID$ : La distance entre les distributions de probabilité des images générées et des images réelles.

Les VAEs ont un bon score $log(P(x))$ mais un mauvais $FID$ et génère donc de grandes données diversifiées.

Les GANs ont un mauvais score $log(P(x))$ mais un bon $FID$ et génère donc des données peu diversifiées mais de bonne qualité (tout en étant difficile à entraîner).

L'objectif est de trouver un entre-deux.

## Architecture

L'architecture générale est similaire à celle d'un VAE :
* Notre donnée en entrée est bruitée via un encodeur.
* Elle est projetée dans l'espace latent $Z$.
* Elle est décodée pour revenir dans l'espace de sortie.

La différence se fait dans la taille de l'espace latent. Pour un VAE, celui-ci était de petite taille ; pour un DPM, il sera de la même dimension que l'espace de sortie. De plus, la phase d'encodage n'utilisera pas de réseau de neurones contrairement aux VAEs, mais un encodeur déterministe.

## Principe
*Définition* : Lorsque l'on parle de bruit sur une image, on parle de bruit gaussien. C'est-à-dire que chaque pixel est tiré aléatoirement suivant une loi noramle centrée réduite.

Il y a trois processus dans un DPM :
* Le Forward Diffusion Process : Qui consiste à ajouter du bruit à notre image de départ pour obtenir une image bruitée. Il va ajouter un bruit un nombre discret de fois : $T$ (un hyperparamètre qui varie de 100 a quelques milliers). $x_0$ est notre image de départ et $x_T$ est notre image bruitée et $x_{t=5}$ est moins bruitée que $x_{t=10}$. À $x_T$ notre image est un pur bruit gaussien.
* Le Reverse Diffusion Process : C'est là que l'on a notre réseau de neurones. Il va prendre une image bruitée $x_t$ en entrée, et il va ressortir une image moins bruitée $x_{t-1}$. 

  L'objectif de notre modèle est qu'il soit capable de débruiter : 
  * Une image de n'importe quel temps $t$ à l'étape précedante $t-1$.
  * N'importe quelle image de notre dataset

* Sampling Process : Maintenant, on est capable, à partir de n'importe quel bruit gaussien $x_T$, de le débruiter pour obtenir une image $x_0$.

## Forward Diffusion Process

Pour passer d'un image $x_{t-1}$ à $x_t$, on va rajouter du bruit gaussien suivant la chaîne de Markov. Le bruit $q$ rajouté suit une loi normale : $q(x_t|x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_tI)$ avec $\sqrt{1-\beta_t}x_{t-1}$ la moyenne et $\beta_tI$ la variance. 

Cela revient à écrire : $x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}z_{t-1}$ avec $z_{t-1} \sim \mathcal{N}(0,I)$ et $\beta_t \in [0,1]$ sachant que $\forall t ~\beta_t < \beta_{t+1}$.

Finalement, il existe une façon d'obtenir $x_t$ à partir de $x_0$ sans générer toutes les images intermédiaires : $x_t = \sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}z $ avec  $\bar{\alpha_t} = \prod_{i=1}^{t}(1-\beta_i)$ et $z \sim \mathcal{N}(0,I)$.

## Reverse Diffusion Process

Comme l'image $x_t$ est bruitée par $z_{t}$ à partir de $x_0$, réussir à retrouver $x_{t-1}$ est équivalent à réussir à trouver le bruit qui a été utilisé. L'objectif pour trouver $x_{t-1}$ est donc de trouver $z_{t}$ à partir de $x_t$.

## Entraînement
Le principe de l'algorithme est le suivant :
* On prend notre image $x_0$.
* On génère aléatoirement $t \in [|0,T|]$.
* On génère notre bruit gaussien (qui a la forme de l'image) $z_t$
* On génère notre image bruitée $x_t$ suivant l'algorithme de forward diffusion : $x_t = \sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}z_t $.
* On passe notre image bruitée dans notre réseau de neurones pour essayer d'obtenir le bruit qui l'a générée : $z_{pred} = f(x_t,t)$.
* On calcul la perte : $loss = ||z_{pred} - z_t||_2^2$.
* On fait la rétropropagation de notre erreur dans notre réseau de neurones.

## Génération d'image (sampling process)
Pour généere une image, le principe est le suivant :
* On génère $x_T$ notre bruit gaussien.
* On rentre dans notre modèle $x_T$ et $T$ afin d'obtenir $z_{pred} = f(x_T,T) \approx z_T$.
* On peut donc calculer $x_{t-1}$ de façon approchée : $x_{t-1}\approx \frac{1}{\sqrt{1-\beta_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha_t}}}z_{pred})$.
* On recommence jusqu'à obtenir $x_0$.

## Améliorations
* **Cosine Scheduling** : L'objectif est de faire en sorte que l'évolution de $\beta_t$ suive une évolution en $cos^2$. Cela permet de rendre l'évolution de $\bar{\alpha_t}$ plus lente et plus linéaire ce qui rend l'impact du bruit mieux réparti lors du débruitage.
* **Variance Learning**

## Noise Intermolation

Une chose importante à noter est que l'on peut faire de l'interpolation de bruit. C'est à dire que l'on peut prendre deux bruits $z_1$ et $z_2$ et en générer un troisième $z_3$ qui est un mélange des deux. Ainsi, si $x_1$ et $x_2$ sont générés par les bruits $z_1$ et $z_2$, alors l'image $x_3$ générée par le bruit $z_3$ sera un mélange des deux images $x_1$ et $x_2$. Ex : $x_1$ est un cheval, $x_2$ est un ange, $x_3$ peut ressembler à pégase.

## Entrainement du text-to-image

L'objectif est d'entrainer notre modèle à transformer un mot en un point dans l'espace latent, puis de transformer nos mots en image, notamment en utilisant de la self attention. Par contre, la génération d'image se fait dirrectement depuis l'espace latent.
