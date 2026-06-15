SCENE_DIR="/home/ubuntu/datasets/ScanNet/processed/scene0000_01"
OUT_DIR="/home/ubuntu/datasets/ScanNet/processed/scene0000_01_colmap"
X=4 # número de frames a saltar

mkdir -p $OUT_DIR/input

c=0
for f in $(ls $SCENE_DIR/color/*.jpg | sort -V); do
  if [ $((c % X)) -eq 0 ]; then
    cp $f $OUT_DIR/input/
  fi
  c=$((c + 1))
done

cd /home/ubuntu/semantic-gaussians
rm -rf $OUT_DIR/distorted
xvfb-run -a uv run python -m core.data.preprocessing.sfm --source-path $OUT_DIR --sfm colmap

ln -s $SCENE_DIR/sam $OUT_DIR/sam

mkdir -p $OUT_DIR/depth
for f in $OUT_DIR/images/*.jpg; do
    filename=$(basename "$f")
    frame_num=${filename%.jpg} 
    ln -s $SCENE_DIR/depth/${frame_num}.png $OUT_DIR/depth/${frame_num}.png
done

echo "¡Completado!"
