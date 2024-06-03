#!/bin/bash

JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home
WEKA_CLASSIFIER_DIR=/Users/rafid/Downloads/Doc/Tubes\ AI/spam-classification-weka-java

cd "$WEKA_CLASSIFIER_DIR"
"$JAVA_HOME/bin/java" --add-opens java.base/java.lang=ALL-UNNAMED WekaClassifier
