# reconhecimento-emoces-CNN

Reconhecimento de emoções em imagens utilizando CNN.


## Dataset

O dataset utilizado foi o [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) disponibilizado no Kaggle. E também foi utilizado o [AffectNet](http://mohammadmahoor.com/affectnet/) para treinamento.

## Pré-processamento

Com os datasets em mãos, foi necessário realizar um pré-processamento para que os dados pudessem ser utilizados para treinamento. O pré-processamento consiste em:

- Conversão das imagens para escala de cinza;
- Conversão das imagens para o formato 48x48;
- Filtro Gaussiano para suavização das imagens;
- Equalização do histograma das imagens;

Para realizar o pré-processamento, foi utilizado o script [pre-processamento.ipynb](https://github.com/Guilherme-Fumagali/reconhecimento-emoces-CNN/blob/main/pre-processamento.ipynb).
Nele, depois de baixado e extraído os datasets, estes devem ser inseridos na pasta `datasets` e executado o script.

