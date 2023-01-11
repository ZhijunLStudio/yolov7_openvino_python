## Installation

```sh
pip install -r requirements.txt
```

## usage

Use the export script in the Yolov7 warehouse to export the onnx model in intermediate format

```sh
python export.py --weights weights/best.pt --fp16
```

Use the mo tool to export the onnx intermediate model to openvino model

```sh
mo --input_model weights/best.onnx --output_dir yolov7_openvino
```

run

```sh
python detect.py
```

