# demo code
python demos/demo_gan.py -i /mnt/sdh/sgraph/DECA/TestSamples/examples --saveDepth True --saveObj True --saveImages True --render_orig 0 --useTex 1 -p --extractTex 1 --cfg /mnt/sdh/sgraph/DECA/configs/release_version/deca_gan.yml
# training code
python main_train.py --cfg ./configs/release_version/deca_gan.yml


# tar model checkpoint
# zip /mnt/sdh/sgraph/DECA/training/gan/4/models/00085000.tar and config.yaml and model.tar in 'model.zip'
zip -r model.zip /mnt/sdh/sgraph/DECA/training/gan/4/models/00085000.tar /mnt/sdh/sgraph/DECA/training/gan/4/config.yaml