cd /content/sequence_classification/data
tar -zxvf /content/drive/MyDrive/data/data.tar.gz 
pip install -r /content/sequence_classification/requirements.txt
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)