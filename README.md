# Projeto_Ensemble_Modelo_Blending_Previsao_Ado-o_de_Pets
Projeto de classificação binária para prever a probabilidade de adoção de animais de um abrigo a partir de características como espécie, raça, idade, porte, saúde e vacinação.Inclui EDA, três abordagens de modelagem e a correção de um caso de data leakage no modelo de stacking.

🎯 Objetivo

Identificar quais fatores mais influenciam a adoção e construir modelos capazes de apontar quais animais têm alta ou baixa chance de serem adotados, apoiando decisões do abrigo (priorização de vacinação, divulgação, definição de taxas).

📂 Estrutura do repositório
Arquivo	Descrição
hands-on_2_1.ipynb	Notebook final: EDA, pré-processamento, modelagem e stacking corrigido
pet_traduzido.csv	Base de dados com colunas traduzidas para o português
Dicionario_de_Dados_-_Adoção_de_Pets.pdf	Dicionário de dados (nomes originais e traduzidos)
relatorio_sweetviz_pet.html	Relatório de EDA gerado pelo notebook (Sweetviz)
📊 Sobre os dados
2.007 registros e 13 colunas, sem valores nulos e sem duplicatas.
Variável-alvo: Probabilidade_Adocao (0 = baixa, 1 = alta), com ~67% na classe 0 e ~33% na classe 1.
Categóricas: Tipo_Pet, Raca, Cor, Porte.
Numéricas: Idade_Meses, Peso_Kg, Dias_No_Abrigo, Taxa_Adocao.
Binárias: Vacinado, Condicao_Saude, Dono_Anterior.
🔬 Metodologia
Inspeção – info(), shape, tipos de dados.
EDA – Pair Plot segmentado pelo alvo e relatório automatizado com Sweetviz.
Pré-processamento – One-Hot Encoding (drop_first=True), padronização com StandardScaler e remoção de ID_Pet.
Modelos individuais (split 80/20, random_state=42): Regressão Logística e Árvore de Decisão (max_depth=5, min_samples_split=10, min_samples_leaf=5).
Stacking (meta-modelo) com Regressão Logística sobre as probabilidades dos dois modelos base, construído sem vazamento de dados (detalhes abaixo).
Avaliação – Acurácia, Precisão, Recall, ROC AUC, matrizes de confusão (treino e teste) e validação cruzada estratificada de 5 folds.
🛠️ Correção do data leakage no stacking

A primeira versão do meta-modelo chegou a 100% em todas as métricas, um sinal clássico de vazamento de dados. A investigação apontou duas causas:

os modelos base foram treinados com toda a base (inclusive as linhas que depois viraram teste) antes de gerar as probabilidades do meta-modelo;
a Árvore de Decisão do blending não tinha limite de profundidade e decorava o alvo.

Correções aplicadas na versão final:

Antes	Depois
Treino dos modelos base com 100% da base	Split treino/teste primeiro, com stratify=y
Probabilidades "vistas" pelo modelo no treino	Probabilidades out-of-fold via cross_val_predict
Árvore sem limite de profundidade	Árvore com max_depth=5 e regularização
StandardScaler ajustado antes do split	Scaler dentro de um Pipeline, ajustado apenas no treino
Avaliação só no meta-modelo manual	Validação adicional com StackingClassifier + CV de 5 folds
📈 Resultados

Conjunto de teste (20%)

Modelo	Acurácia	Precisão	Recall	ROC AUC
Regressão Logística	0,908	0,936	0,773	0,948
Árvore de Decisão	0,950	0,931	0,917	0,951
Meta-modelo (stacking corrigido)	0,930	0,919	0,864	0,921

Validação cruzada (5 folds)

Modelo	Acurácia	ROC AUC
Regressão Logística	0,907 ± 0,008	0,928 ± 0,022
Árvore de Decisão	0,939 ± 0,015	0,923 ± 0,023
Stacking	0,940 ± 0,015	0,929 ± 0,022

Conclusões:

A Árvore de Decisão é o melhor equilíbrio entre desempenho e simplicidade: maior acurácia e, principalmente, maior recall (0,92 vs 0,77 da Regressão Logística).
O stacking corrigido não trouxe ganho relevante sobre a Árvore: na validação cruzada, os dois ficam praticamente empatados (0,940 vs 0,939 de acurácia).
Treino e teste têm valores próximos em todos os modelos, sem sinais de overfitting.
A correção do vazamento reduziu o meta-modelo de 100% para ~93%, que é o desempenho real.
