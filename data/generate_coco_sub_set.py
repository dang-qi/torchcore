import json
def generate_json_by_id(valid_ids, input_path, out_path):
    with open(input_path, 'r') as f:
        result = json.load(f)
        #result.keys = dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
        # select the proper iamges
        out_images = [im for im in result['images'] if im['id'] in valid_ids]
        print(out_images)
        result['images'] = out_images

        # select the proper annotations
        out_annos = [anno for anno in result['annotations'] if anno['image_id'] in valid_ids]
        print(out_annos)
        result['annotations'] = out_annos

    out_f = open(out_path, 'w')
    json_string = json.dumps(result)
    json.dump(json_string,out_f)
    print('out file is saved to {}'.format(out_path))

if __name__=='__main__':
    valid_ids = [32811]
    input_path = 'data/coco/annotations/instances_val2017.json'
    out_path = 'data/coco/annotations/intstances_val2017_out.json'
    generate_json_by_id(valid_ids, input_path, out_path)
