import sys
from rknn.api import RKNN

def convert_onnx_to_rknn():
    """Convert ONNX model to RKNN format."""
    # Initialize RKNN object
    rknn = RKNN(verbose=True)

    # Configure model for target platform, you might need to tweak platform config Here
    print('--> Configuring model')
    rknn.config(target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(
        model='/home/local/train/BookShelf/lfm2/lfm_model_float32.onnx',
        input_size_list=[[1, 128], [1, 128], [1, 128]],
        inputs=['input_ids', 'attention_mask', 'position_ids']
    )
    if ret != 0:
        print('Load model failed!')
        sys.exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        sys.exit(ret)
    print('done')

    # Export RKNN model
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn("./lfm2-350.rknn")
    if ret != 0:
        print('Export RKNN model failed!')
        sys.exit(ret)
    print('done')

    # Release RKNN resources
    rknn.release()

if __name__ == '__main__':
    convert_onnx_to_rknn()
