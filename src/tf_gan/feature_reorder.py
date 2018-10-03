""" module to get the desierble order of features  """

import numpy as np

feature_name_celeba_org = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
    'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

feature_name_celeba_reorder = [
    'Male', 'Young', 'Pale_Skin', 'Heavy_Makeup',
    'Receding_Hairline', 'Bangs', 'Straight_Hair', 'Wavy_Hair', 'Bald',
    'Arched_Eyebrows', 'Bushy_Eyebrows', 'Narrow_Eyes', 'Bags_Under_Eyes',
    'Big_Nose', 'Pointy_Nose', 'High_Cheekbones', 'Rosy_Cheeks',
    'Mouth_Slightly_Open', 'Smiling', 'Big_Lips', 'Wearing_Lipstick',
    'No_Beard', 'Mustache', 'Goatee', 'Sideburns',
    'Black_Hair', 'Brown_Hair', 'Blond_Hair', 'Gray_Hair',
    'Chubby', 'Oval_Face', 'Double_Chin',
    'Eyeglasses', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Necktie', 'Wearing_Necklace',
    '5_o_Clock_Shadow', 'Blurry', 'Attractive',
]

