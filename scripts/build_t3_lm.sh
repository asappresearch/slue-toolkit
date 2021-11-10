if ! [ -d data/TEDLIUM_release-3 ]; then
    mkdir -p data
    cd data
    if ! [ -f TEDLIUM_release-3.tgz ]; then
        echo "download tedlium 3"
        wget https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz -O TEDLIUM_release-3.tgz 
    fi
    echo "extract tedlium 3"
    tar -zxvf TEDLIUM_release-3.tgz TEDLIUM_release-3/LM
    cd ..
fi

echo "combine lm corpus"
if ! [ -f data/TEDLIUM_release-3/LM/all_text.en ]; then
    cat data/TEDLIUM_release-3/LM/*.gz > data/TEDLIUM_release-3/LM/all_text.en.gz
    gunzip data/TEDLIUM_release-3/LM/all_text.en.gz
fi

echo "create ngram LM"
bash scripts/create_ngram.sh $1 data/TEDLIUM_release-3/LM/all_text.en save/kenlm/t3 3

