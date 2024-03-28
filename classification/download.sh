wget -O classification_models.zip https://drive.switch.ch/index.php/s/Mar9rweWDSxEL7k/download
unzip classification_models.zip
mkdir -p pretrained_models
mv classification pretrained_models
rm classification_models.zip