# stone-entropy-watermark

Marca d'água em código gerado por LLM: testa se filtrar por entropia (ideia do SWEET)
sobre os tokens não sintáticos do STONE melhora a detecção e preserva a corretude do código.

## Pergunta de pesquisa

O STONE marca apenas tokens não sintáticos, mas não considera a incerteza do modelo em cada
posição. O SWEET marca apenas tokens de alta entropia, mas inclui tokens sintáticos.
Este projeto investiga se combinar os dois critérios traz ganhos em relação a cada um:

- na **detectabilidade** (AUROC, z-score);
- na **corretude** do código gerado (pass@1).

## Estado atual

Piloto exploratório, em andamento. Os notebooks medem a entropia nas posições em que o STONE
marca e o sinal de detecção com um filtro de entropia aplicado depois da geração.
A geração com o filtro incluído e a medição de pass@1 ainda não foram feitas.
Resultados preliminares, com amostras pequenas, não permitem concluir que o método combinado
é melhor.

## Notebooks (`notebook/`)

| Arquivo | O que faz |
|---|---|
| `stone_entropy_pilot_v3.ipynb` | Piloto no HumanEval+ (modos `repo` e `paper`), com truncagem e testes pareados |
| `stone_entropy_pilot_v4.ipynb` | Distribuição de entropia nas posições que o STONE conta |
| `stone_entropy_pilot_v5.ipynb` | Comprimento, entropia e detecção em HumanEval+, APPS e ClassEval |

Foram escritos para o Google Colab (GPU T4). Cada um tem instruções no início.

## Dados

- [HumanEval+](https://github.com/evalplus/humanevalplus_release)
- [APPS](https://huggingface.co/datasets/codeparrot/apps) (somente tarefas com função)
- [ClassEval](https://huggingface.co/datasets/FudanSELab/ClassEval)

## Modelo

Qwen2.5-Coder-3B (modo `paper`) e 3B-Instruct (modo `repo`).

## Referências

- STONE: Kim et al., *Marking Code Without Breaking It: Code Watermarking for Detecting
  LLM-Generated Code* (2026). arXiv:2502.18851
- SWEET: Lee et al., *Who Wrote this Code? Watermarking for Code Generation* (ACL 2024).
  arXiv:2305.15060

## Autoria

Leonardo Lima, Lívia Maria Rocha e Maria Eduarda Souza — INF/UFG.
