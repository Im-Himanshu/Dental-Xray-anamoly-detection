import json

full_json = {}

expert_label = json.load(open('./Expert/expert.json'))
#expert_label[43]['Label']['objects']

for value in expert_label:
    image_name = value['External ID']
    label = value['Label']
    oral_description = value['Description']
    objects = label['objects']
    regions = []
    #classifications = label['classifications']
    for object in objects:
        classifications = object['classifications']
        polygons = object['polygons']

        region_attributes = []
        for classification in classifications:
            results= {}
            title = classification['title']
            value = classification['value']
            answer = classification['answer']
            del answer['featureId']
            del answer['schemaId']
            if not value in results:
                results[value] = answer

            elif answer['title'] is not None: #update with lowest one, may be buggy
                results[value] = answer
            region_attributes.append(results)


        for polygon in polygons:
            region = {"shape_attributes": {
                "name": "polygon",
                "all_points_x": None,
                "all_points_y": None},
                "region_attributes": region_attributes}
            regions.append(region)
            region["all_points_x"], region["all_points_x"] = all_x, all_y = [], []
            for x,y in polygon:
                all_x.append(x)
                all_y.append(y)




    image_label = full_json[image_name] = {}
    image_label['expert_label'] = value

count =1
for value in expert_label:
    count+=1
    image_name = value['External ID']
    if len(value['Label']['objects'])>1:
        print(image_name, count, len(value['Label']['objects']))

print("something")