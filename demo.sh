python demos/demo_gan.py \
    -i TestSamples/ffhq \
    --saveDepth True \
    --saveObj True \
    --saveImages True \
    --render_orig 0 \
    --useTex 1 \
    -p \
    --extractTex 1 \
    --cfg ./configs/release_version/deca_gan.yml

# demo 
python demos/demo_reconstruct.py \
    -i TestSamples/ffhq \
    --saveDepth True \
    --saveObj True \
    --saveImages True \
    --render_orig 0 \
    --useTex 1 \
    -p \
    --extractTex 1 \
    --cfg /mnt/sdh/sgraph/DECA/configs/release_version/deca_coarse.yml
