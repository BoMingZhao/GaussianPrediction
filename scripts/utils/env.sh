conda install -c nvidia cudatoolkit=11.7
conda install -c conda-forge cudatoolkit-dev=11.7

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

pip install git+https://github.com/BoMingZhao/tiny-cuda-nn-float32/#subdirectory=bindings/torch
pip install submodules/diff-gaussian-rasterization-w-depth
pip install submodules/simple-knn
pip install submodules/FRNN/external/prefix_sum
pip install submodules/FRNN
cd submodules/lib/pointops
pip install .
pip install "git+https://github.com/facebookresearch/pytorch3d.git"