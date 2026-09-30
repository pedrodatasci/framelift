<div align="center">

# 🎬 Galeria do framelift

**Exemplos reais de antes e depois.**

[← Voltar ao README](README.pt-BR.md) · [English 🇺🇸](GALLERY.md)

</div>

Cada exemplo mostra a mesma área lado a lado: o original à esquerda (só esticado para o
mesmo tamanho) e o framelift à direita. O primeiro GIF é um recorte em 100% da área mais
detalhada; abra **Frame inteiro** para ver a imagem completa.

<sub>Os GIFs são comprimidos e reduzidos para esta página; o MP4 real gerado é mais nítido.</sub>

## Sumário

- [Show de rock ao vivo](#show-de-rock-ao-vivo)
- [Clipe musical colorido](#clipe-musical-colorido)
- [Gravação de show em preto e branco](#gravação-de-show-em-preto-e-branco)
- [Adicione os seus](#adicione-os-seus)

## Show de rock ao vivo

Movimento rápido, cortes seguidos e luz de palco piscando: o tipo de vídeo mais difícil desta
galeria. Os rostos da plateia ficam mais definidos e os blocos de compressão somem. O borrão
de movimento continua, porque nenhum upscaler consegue desfazê-lo. 480p → 1080p.

![Antes e depois: show de rock ao vivo, com zoom na plateia](docs/gallery/live_show_zoom.gif)

<details>
<summary>Frame inteiro</summary>

![Antes e depois: show de rock ao vivo, frame inteiro](docs/gallery/live_show_full.gif)

</details>

## Clipe musical colorido

Cordas e trastes mais limpos, e menos blocos de compressão. 480p → 1080p.

![Antes e depois: clipe musical colorido, com zoom](docs/music_video_zoom.gif)

<details>
<summary>Frame inteiro</summary>

![Antes e depois: clipe musical colorido, frame inteiro](docs/music_video_full.gif)

</details>

## Gravação de show em preto e branco

Granulação e blocos de compressão limpos, com bordas mais nítidas nos rostos e nos óculos.
480p → 1080p, processado na CPU de um notebook.

![Antes e depois: gravação de show em preto e branco, com zoom](docs/demo_zoom.gif)

<details>
<summary>Frame inteiro</summary>

![Antes e depois: gravação de show em preto e branco, frame inteiro](docs/demo_full.gif)

</details>

## Adicione os seus

Os GIFs desta página são gerados por um script do próprio repo. Rode a partir da pasta raiz
do repo, com o vídeo original e a saída do framelift para ele:

```bash
python scripts/make_comparison.py original.mp4 enhanced.mp4 --name meu_clipe
```

Ele escolhe a cena estável mais nítida (movimento rápido e cortes deixam o zoom pulando),
aproxima na área com mais detalhe, reduz a taxa de quadros e o tamanho até cada GIF ficar
abaixo de 6 MB, e mostra o Markdown para colar aqui. Ele grava em `docs/gallery/`: dois GIFs
para commitar e, de brinde, os mesmos dois em MP4 com qualidade total para compartilhar (o
`.gitignore` deixa esses de fora do git).

Se preferir escolher o zoom você mesmo, `--zoom X,Y,W,H` define a região (em pixels do vídeo
original) e `--start`/`--end` definem a cena (em segundos).
