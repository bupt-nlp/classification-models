# text classification models

Out of box classification models based on PaddlePaddle Deeplearning Freamework.

We will implemented text classification models in one-style code. If you want to cutomize model, all you need is to write your Network, and you can run it.  

To achieve the above goal, you should keep you corpus is consist with the following data structure:

```text
{"text": "", "text_pair": "", "label": ""}
{"text": "", "text_pair": "", "label": ""}
...
```

**We are welcome to Post issue for new models, or pr to contribute.**

## Model List

- [x] [Pretrained Model + Linear Model](./src/models/pretrained.py)
- [ ] [CNN](./src/models/cnn.py)
- [ ] [RNN](./src/models/rnn.py)

## Corpus

- [ ] ATIS
- [ ] SNIPS
- [ ] CLINC150
- [ ] AGNews 

## Experiments

- [ ] TODO： add experiment metrics at here

## Creators

- [@wj-Mcat](https://github.com/wj-Mcat) - Jingjing WU (吴京京)

## Copyright & License

- Code & Docs © 2022 wj-Mcat
- Code released under the Apache-2.0 License
- Docs released under Creative Commons
