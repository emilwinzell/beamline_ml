import xml.etree.ElementTree as ET
import pandas as pd
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    labels = os.path.join(args.base,'data.xml')

    tree = ET.parse(labels)
    root = tree.getroot()

    data_df = pd.DataFrame(columns=('sample', 'pitch','yaw','roll'))

    for item in root:
        if item.tag[:6] == 'sample':
            samplenr = int(item.tag[7:])
            for i in item:
                
    
    return




if __name__ == '__main__':
    main()