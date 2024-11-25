git clone -n --depth=1 --filter=tree:0 https://github.com/mvaldenegro/marine-debris-fls-datasets.git marine_debris_data
cd marine_debris_data
git sparse-checkout set --no-cone "md_fls_dataset/data/watertank-segmentation/BoxAnnotations" "md_fls_dataset/data/watertank-segmentation/Images"
git checkout
mv md_fls_dataset/data/watertank-segmentation/* ./
rm -r md_fls_dataset
mv BoxAnnotations annotations
mv Images images
cd ..

echo RENAMING
python data_tools.py marine_debris_data --rename

echo SCALING
python data_tools.py marine_debris_data --scale
