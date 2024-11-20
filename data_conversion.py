import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def convert_to_voc_format(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        x = int(bndbox.find('x').text)
        y = int(bndbox.find('y').text)
        w = int(bndbox.find('w').text)
        h = int(bndbox.find('h').text)

        xmin, ymin = x, y
        xmax, ymax = x + w, y + h
        objects.append({'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    return objects

def save_as_voc_format(objects, output_file):
    # Create the root XML structure
    annotation = ET.Element('annotation')

    for obj in objects:
        object_element = ET.SubElement(annotation, 'object')
        name_element = ET.SubElement(object_element, 'name')
        name_element.text = obj['name']

        bndbox_element = ET.SubElement(object_element, 'bndbox')
        xmin_element = ET.SubElement(bndbox_element, 'xmin')
        xmin_element.text = str(obj['xmin'])
        ymin_element = ET.SubElement(bndbox_element, 'ymin')
        ymin_element.text = str(obj['ymin'])
        xmax_element = ET.SubElement(bndbox_element, 'xmax')
        xmax_element.text = str(obj['xmax'])
        ymax_element = ET.SubElement(bndbox_element, 'ymax')
        ymax_element.text = str(obj['ymax'])

    # Pretty print and write to file
    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    with open(output_file, "w") as f:
        f.write(xmlstr)

# Convert and save all annotations in a folder
def process_annotations(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for annotation_file in os.listdir(input_dir):
        if annotation_file.endswith('.xml'):
            input_path = os.path.join(input_dir, annotation_file)
            output_path = os.path.join(output_dir, annotation_file)

            objects = convert_to_voc_format(input_path)
            save_as_voc_format(objects, output_path)
            print(f"Processed and saved: {annotation_file}")

# Example usage
# input_dir = "./marine_debris_data/annotations"  # Replace with the directory containing your annotation XML files
# output_dir = "newAnnotations"  # Replace with the desired output directory
# process_annotations(input_dir, output_dir)
