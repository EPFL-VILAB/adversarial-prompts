
from .textual_inversion_dataset import TextualInversionDataset as TI_taskonomy
from .textual_inversion_dataset_omnidata import TextualInversionDataset as TI_omnidata

def get_dataset(
        dataset,
        cond_data_root,
        mask_data_root,
        tokenizer,
        size=512,
        pred_size=384,
        repeats=100,
        center_crop=False,
        negative_prompt="",
        num_new_tokens=4,
        data_paths_file=None,
        add_default_prompt=False,
        taskonomy_split_path=None,
    ):

    if dataset == 'taskonomy':
        dataset = TI_taskonomy(
        cond_data_root=cond_data_root,
        mask_data_root=mask_data_root,
        tokenizer=tokenizer,
        size=size,
        pred_size=pred_size,
        repeats=repeats,
        center_crop=center_crop,
        negative_prompt=negative_prompt,
        num_new_tokens=num_new_tokens,
        data_paths_file=data_paths_file,
        add_default_prompt=add_default_prompt,
        taskonomy_split_path=taskonomy_split_path
        )

    else:
        dataset = TI_omnidata(
        tokenizer=tokenizer,
        size=size,
        pred_size=pred_size,
        repeats=repeats,
        center_crop=center_crop,
        negative_prompt=negative_prompt,
        num_new_tokens=num_new_tokens,
        data_paths_file=data_paths_file,
        add_default_prompt=add_default_prompt,
        )

    return dataset




