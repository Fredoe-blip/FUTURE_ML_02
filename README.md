\#  Classification automatique de tickets support



\[!\[Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

\[!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)

\[!\[F1 Score](https://img.shields.io/badge/F1--Score-0.846-green.svg)]()



\##  Objectif



Classifier automatiquement des tickets support en \*\*8 catégories techniques\*\* à partir de leur description textuelle.



\##  Performances



| Modèle | F1-score |

|--------|----------|

| \*\*Logistic Regression + SMOTE\*\* | \*\*0.846\*\* |

| Random Forest | 0.842 |

| Logistic Regression | 0.840 |



\##  Installation



```bash

git clone https://github.com/Fredoe-blip/FUTURE\_ML\_02.git

cd FUTURE\_ML\_02

pip install -r requirements.txt

python -m spacy download en\_core\_web\_sm

