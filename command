python EdgeaiYolox/tools/train.py -f EdgeaiYolox/exps/example/custom/yolox_m_myExp_1280.py -d 1 -b 8 --fp16 -o -c EdgeaiYolox/yolox_m_weights.pth

python EdgeaiYolox/tools/export_onnx.py -f EdgeaiYolox/exps/example/custom/yolox_m_myExp_1280.py -c EdgeaiYolox/YOLOX_outputs/yolox_m_myExp/best_ckpt.pth --output yolox_m_pose.onnx --export-det --opset 11 --dataset custom --task object_pose



