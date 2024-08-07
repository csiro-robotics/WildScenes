_base_ = ["wildscenes.py"]
custom_label_map = {"unlabelled": "unlabelled",
    "asphalt/concrete": "other-terrain",
    "dirt": "dirt",
    "mud": "mud",
    "water": "water",
    "gravel": "gravel",
    "other-terrain": "other-terrain",
    "tree-trunk": "tree-trunk",
    "tree-foliage": "tree-foliage",
    "bush": "bush",
    "grass": "grass",
    "fence": "fence",
    "other-structure": "structure",
    "pole": "object",
    "vehicle": "unlabelled",
    "rock": "rock",
    "log": "log",
    "other-object": "object",
    "sky": "sky"}
val_custom_label_map = {"unlabelled": "unlabelled",
    "asphalt/concrete": "unlabelled",
    "dirt": "dirt",
    "mud": "unlabelled",
    "water": "water",
    "gravel": "gravel",
    "other-terrain": "unlabelled",
    "tree-trunk": "tree-trunk",
    "tree-foliage": "tree-foliage",
    "bush": "bush",
    "grass": "grass",
    "fence": "unlabelled",
    "other-structure": "structure",
    "pole": "object",
    "vehicle": "unlabelled",
    "rock": "rock",
    "log": "log",
    "other-object": "object",
    "sky": "sky"}
num_classes=15
# Add the custom label map to all of the dataloaders
train_dataset = dict(custom_label_map=custom_label_map)
train_dataloader = dict(dataset=train_dataset)
val_dataset = dict(custom_label_map=custom_label_map)
val_dataloader = dict(dataset=val_dataset)
test_dataset = dict(custom_label_map=custom_label_map)
test_dataloader = dict(dataset=test_dataset)
