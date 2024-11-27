wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar
rm datasets.tar

mv original/SST-2 original/SST-2-original
mv original/GLUE-SST-2 original/SST-2

python generate_data.py --k 512 --save_dir k-data