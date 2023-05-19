import torch
import os.path as osp, os
import onnxruntime
from x_dgcnn import DGCNN_Cls, DGCNN_Seg

save_dir = "checkpoints"


def export_and_check_dgcnn_cls():
    model = DGCNN_Cls(k=40, in_dim=3, out_dim=40)
    x = torch.randn(20, 2048, 3)
    xyz = x
    inputs = {"x": x,
              "xyz": xyz,
              }
    dynamic_axes = {"x": {0: "batch_size", 1: "num_points"}, "xyz": {0: "batch_size", 1: "num_points"}}

    # export
    os.makedirs(save_dir, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        osp.join(save_dir, "dgcnn_cls.onnx"),
        verbose=False,
        opset_version=14,
        dynamic_axes=dynamic_axes,
        input_names=["x", "xyz"],
        output_names=["output"]
    )

    # check with dynamic shape
    sess = onnxruntime.InferenceSession(osp.join(save_dir, "dgcnn_cls.onnx"))
    x = torch.randn(9, 2048, 3)
    xyz = x
    pred = sess.run(['output'], {'x': x.numpy(), "xyz": xyz.numpy()})[0]
    print("success, the output shape is {}".format(pred.shape))


def export_and_check_dgcnn_seg():
    model = DGCNN_Seg(k=20, in_dim=3, out_dim=10, n_category=100)
    x = torch.randn(20, 2048, 3)
    xyz = x
    category = torch.randint(0, 100, (20,))
    inputs = {
        "x": x,
        "xyz": xyz,
        "category": category
    }

    os.makedirs(save_dir, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        osp.join(save_dir, "dgcnn_seg.onnx"),
        verbose=False,
        opset_version=14,
        dynamic_axes={"x": {0: "batch_size", 1: "num_points"},
                      "xyz": {0: "batch_size", 1: "num_points"},
                      "category": {0: "batch_size"}},
        input_names=["x", "xyz", "category"],
        output_names=["output"]
    )

    # to onnxruntime
    sess = onnxruntime.InferenceSession(osp.join(save_dir, "dgcnn_seg.onnx"))
    x = torch.randn(9, 2048, 3)
    xyz = x
    category = torch.randint(0, 100, (9,))
    pred = sess.run(["output"], {'x': x.numpy(), "xyz": xyz.numpy(), "category": category.numpy()})[0]
    print("success, the output shape is {}".format(pred.shape))


if __name__ == '__main__':
    export_and_check_dgcnn_cls()
    export_and_check_dgcnn_seg()
