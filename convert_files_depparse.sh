#!/bin/bash
chmod +x convert_files.sh
FILE_ARRAY=('../project/L545-B659-Final-Project/StanceDataset/tweet_textfiles/*')
for i in ${FILE_ARRAY[*]}
do
java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,depparse -file $i -outputFormat conll # may want to add ner between lemma and depparse, but it slows the program down a lot
done
OUTPUT_ARRAY=('*.txt.conll')
for filename in ${OUTPUT_ARRAY[*]}; do mv "$filename" "$(echo "$filename" | sed -e 's/.txt/_depparse/g')"; done
