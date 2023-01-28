# demo code
python demos/demo_gan.py -i /mnt/sdh/sgraph/DECA/TestSamples/examples --saveDepth True --saveObj True --saveImages True --render_orig 0 --useTex 1 -p --extractTex 1 --cfg /mnt/sdh/sgraph/DECA/configs/release_version/deca_gan.yml
# training code
python main_train.py --cfg ./configs/release_version/deca_gan.yml