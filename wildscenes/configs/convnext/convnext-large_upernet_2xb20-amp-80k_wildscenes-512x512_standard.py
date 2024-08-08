_base_ = [
        "convnext-large_upernet_2xb20-amp-80k_wildscenes-512x512_template.py", "../_base_/datasets/wildscenes_standard.py"
        ]

num_classes = _base_.num_classes
crop_size = _base_.crop_size
data_preprocessor = dict(size=crop_size)
decode_head = dict(num_classes=num_classes)
aux_head = dict(num_classes=num_classes)
# test_cfg = dict(crop_size=crop_size)
test_cfg = dict(type='TestLoop')
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=decode_head,
    auxiliary_head=aux_head,
    test_cfg=test_cfg
)
