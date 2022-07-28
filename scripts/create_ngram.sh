kenlm=$1 # e.g. /home/username/kenlm/build/bin
input=$2 # input text file
output=$3 # output folder
n=$4 # n gram
mkdir -p $output


arpa=${output}/${n}gram.arpa
bin=${output}/${n}gram.bin


if [[ $input == *"ner"* ]]; then
    cat $input | ${kenlm}/lmplz --skip_symbols -o ${n} > $arpa
else
    cat $input | tr '[A-Z]' '[a-z'] | ${kenlm}/lmplz --skip_symbols -o ${n} > $arpa
fi
${kenlm}/build_binary $arpa $bin

echo 'create lexicon'
python -m slue_toolkit.prepare.create_lexicon $input ${output}/lexicon.lst
rm -f $arpa