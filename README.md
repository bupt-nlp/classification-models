# text classification models

Out of box classification models based on PaddlePaddle Deeplearning Freamework.

We will implemented text classification models in one-style code. If you want to cutomize model, all you need is to write your Network, and you can run it.  

To achieve the above goal, you should keep you corpus is consist with the following data structure:

```text
{"text": "", "text_pair": "", "label": ""}
{"text": "", "text_pair": "", "label": ""}
...
```

## Model List

* [Pretrained Model + Linear Model](./src/models/pretrained.py)
* [CNN](./src/models/cnn.py)
* [RNN](./src/models/rnn.py)
* []()

## Papers

* 