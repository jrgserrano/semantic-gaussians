OUT_DIR="/home/ubuntu/datasets/Replica/office4_colmap"
SCENE_DIR="/home/ubuntu/datasets/Replica/office4"

ln -s $SCENE_DIR/sam $OUT_DIR/sam


mkdir -p $OUT_DIR/depth
for f in $OUT_DIR/images/frame*.jpg; do
    filename=$(basename "$f")
    frame_num=${filename:5:6} 
    ln -s $SCENE_DIR/results/depth${frame_num}.png $OUT_DIR/depth/frame${frame_num}.png
done
