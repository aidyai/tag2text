# --------------------------------------------------------
# The Tag2Text Model
# Copyright (c) 2023 
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinyu Huang
# --------------------------------------------------------

import torch


def inference_tag2text(image, model, input_tag="None"):

    with torch.no_grad():
        caption, tag_predict = model.generate(image,
                                              tag_input=None,
                                              max_length=50,
                                              return_tag_predict=True)

    if input_tag == '' or input_tag == 'none' or input_tag == 'None':
        return tag_predict[0], None, caption[0]

    # If user input specified tags:
    else:
        input_tag_list = []
        input_tag_list.append(input_tag.replace(',', ' | '))

        with torch.no_grad():
            caption, input_tag = model.generate(image,
                                                tag_input=input_tag_list,
                                                max_length=50,
                                                return_tag_predict=True)

        return tag_predict[0], input_tag[0], caption[0]


