'''Question template for question generation'''
yes_no_templates = [{'question':'Is this a {garment_type}','keywords':['garment_type']},
                    {'question':'Is this {garment_type} {color}', 'keywords':['garment_type','color']},
                    {'question':'Is this a {garment_type} with {attribute}', 'keywords':['garment_type','graphical_appearance_name']},
                    {'question':'','keywords':[]}]

other_templates = [{'question':'What kind of garment is this?', 'keywords':[], 'answer':'{garment_type}', 'answer_keywords':['garment_type'],'question_type':'what'},
                   {'quesiton':'What is the color the {}', 'keywords':['garment_type'], 'answer':'{color}', 'answer_keywords':['color'],'question_type':'what'}]