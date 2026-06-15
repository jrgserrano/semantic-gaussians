SCENE_DIR="/home/ubuntu/datasets/Replica/office4"
OUT_DIR="/home/ubuntu/datasets/Replica/office4_colmap"
X=4 # número de frames a saltar

mkdir -p $OUT_DIR/input

c=0
for f in $(ls $SCENE_DIR/results/frame*.jpg | sort); do
  if [ $((c % X)) -eq 0 ]; then
    cp $f $OUT_DIR/input/
  fi
  c=$((c + 1))
done

cd /home/ubuntu/semantic-gaussians
rm -rf /home/ubuntu/datasets/Replica/office4_colmap/distorted
xvfb-run -a uv run python -m core.data.preprocessing.sfm --source-path $OUT_DIR --sfm colmap
