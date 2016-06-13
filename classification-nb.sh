#!/usr/bin/env bash

TEMP_DIR=/tmp/classification-mahout

echo "We will train a Naive Bayes model using Hadoop in pseudo-distributed mode"

mkdir -p ${TEMP_DIR}
echo "Ouput is being directed at ${TEMP_DIR}"

if [ ! -f ${TEMP_DIR}/20news-bydate.tar.gz ]; then
  echo "Downloading 20news-bydate"
  curl http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz -o ${TEMP_DIR}/20news-bydate.tar.gz
  mkdir -p ${TEMP_DIR}/20news-bydate
  echo "Extracting..."
  cd ${TEMP_DIR}/20news-bydate && tar zxf ../20news-bydate.tar.gz
fi

cd $TEMP_DIR

echo "Putting data together"
rm -rf ${TEMP_DIR}/20news-all
mkdir ${TEMP_DIR}/20news-all
cp -R ${TEMP_DIR}/20news-bydate/*/* ${TEMP_DIR}/20news-all

echo "Generating sequence files"
mahout seqdirectory -i ${TEMP_DIR}/20news-all -o ${TEMP_DIR}/20news-seq -ow

echo "Generating bags of words"
mahout seq2sparse -i ${TEMP_DIR}/20news-seq -o ${TEMP_DIR}/20news-vectors -lnorm -nv -wt tfidf

echo "Spliting bags of words in 70-30 for training-testing"
mahout split -i ${TEMP_DIR}/20news-vectors/tfidf-vectors \
  --trainingOutput ${TEMP_DIR}/20news-train-vectors \
  --testOutput ${TEMP_DIR}/20news-test-vectors  \
  --randomSelectionPct 40 --overwrite --sequenceFiles -xm sequential

echo "Training model using Naive Bayes"
mahout trainnb -i ${TEMP_DIR}/20news-train-vectors -o ${TEMP_DIR}/model -li ${TEMP_DIR}/labelindex -ow

echo "Testing model using the test set"
mahout testnb \
  -i ${TEMP_DIR}/20news-test-vectors\
  -m ${TEMP_DIR}/model \
  -l ${TEMP_DIR}/labelindex \
  -ow -o ${TEMP_DIR}/20news-testing
