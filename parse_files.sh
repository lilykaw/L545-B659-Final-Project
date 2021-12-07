#!/bin/bash
chmod +x parse_files.sh
CONNL_ARRAY=('../project/L545-B659-Final-Project/tweet_conll/CoreNLP_Output/'*.conll)
for i in ${CONNL_ARRAY[*]}
do
j=${i/'../project/L545-B659-Final-Project/tweet_conll/CoreNLP_Output/'/}
echo $i
echo ${j/'.conll'/}
#java -jar maltparser-1.9.2.jar -c test -i $i -m parse -o $j'_parsed.conllu' 

#ALGORITHM-OPTIONS - uncomment to try a different one
#default algorithm: nivreeager
#java -jar maltparser-1.9.2.jar -c engmalt.linear-1.7 -i $i -o $j -m parse
#alternative algorithm: stackproj
java -jar maltparser-1.9.2.jar -c engmalt.linear-1.7 -i $i -o $j -m parse -a stackproj

mv "$j" "$(echo "$j" | sed -e 's/.conll/_parsed.conll/g')"
done
