#!/bin/bash
chmod +x convert_files.sh
FILE_ARRAY=('../project/L545-B659-Final-Project/StanceDataset/tweet_textfiles/*txt')
for i in ${FILE_ARRAY[*]}
do
java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma -file $i -outputFormat conll
done
#OUTPUT_ARRAY=('*.txt.conllu')
OUTPUT_ARRAY=('*.txt.conll')
for filename in ${OUTPUT_ARRAY[*]}; do mv "$filename" "$(echo "$filename" | sed -e 's/txt.//g')";  done

