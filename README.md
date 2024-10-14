# TrOCR & CRAFT: Multiline Handwritten Text Recognition (HTR)

This is a simple example code that demonstrate a way to get multiline handwritten text recognition ability for [TrOCR](https://huggingface.co/docs/transformers/en/model_doc/trocr) using [CRAFT](https://arxiv.org/abs/1904.01941) (To detect text regions). This code uses modified code from [craft-text-detector](https://github.com/fcakyon/craft-text-detector) package, which support latest PyTorch versions. 

**Huggingface Model Links:**
- [Small Model](https://huggingface.co/microsoft/trocr-small-handwritten)
- [Larger Model](https://huggingface.co/microsoft/trocr-large-handwritten)

**Requirements:**
- Python 3.12.4
- [PyTorch 2.4.0](https://pytorch.org/get-started/locally)
- gdown 5.2.0
- opencv-python 4.10.0.84
- protobuf 5.28.0
- sentencepiece 0.2.0
- transformers 4.44.2

### Running the example

1. Clone the repository
```
git clone <repo_url>
```
2. Install PyTorch
```
https://pytorch.org/get-started/locally
```
3. Install the requirements
```
pip install -r requirements.txt
```
4. Run
```
python read.py
```
