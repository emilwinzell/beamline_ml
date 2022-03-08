import argparse
import os
import xml.etree.ElementTree as ET

def _indent(elem, level=0):
    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            _indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    xml = os.path.join(args.base,'data.xml')

    tree = ET.parse(xml)
    root = tree.getroot()

    for item in root:
        if item.tag[:6] == 'sample':
            for subitem in item:
                if subitem.tag == 'specifications':
                    s = subitem.find('center_transl')
                    text = s.text.split(':')
                    x = text[1].split(',')[0]
                    z = text[2]
                    subitem.remove(s)
                    xT = ET.SubElement(subitem,'x_transl',{'unit':'mm'})
                    xT.text = x
                    zT = ET.SubElement(subitem,'z_transl',{'unit':'mm'})
                    zT.text = z
                    
    
    tree = ET.ElementTree(_indent(root))
    tree.write(xml, xml_declaration=True, encoding='utf-8')



if __name__ == '__main__':
    main()