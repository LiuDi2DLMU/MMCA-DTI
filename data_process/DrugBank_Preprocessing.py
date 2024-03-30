"""
此文件用于提取drugbank数据库原始数据
"""

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import csv

html = '{http://www.drugbank.ca}'
# here is the location which drugBank xml file is
tree = ET.parse('../data/drugbank/location')
drugs = tree.getroot()


def write_csv(drug_id, data, file_path):
    with open(file_path, "a") as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow([drug_id, data])


def write_file(content, file_path):
    with open(file_path, "a") as file:
        file.write(content)


i = 0
target_ids = []
for drug in drugs:
    if drug.get("type") == "biotech":
        continue
    # drugbank-id
    drug_id = None
    drugbank_ids = drug.findall(html + "drugbank-id")
    for drugbank_id in drugbank_ids:
        if "primary" in drugbank_id.attrib.keys():
            drug_id = drugbank_id.text
    # SMILES  -- calculated-properties
    properties = drug.find(html + "calculated-properties")
    drug_SMILES = None
    drug_InChI = None
    for property in properties:
        if property[0].tag == html + "kind" and property[0].text == "SMILES":
            drug_SMILES = property[1].text
        if property[0].tag == html + "kind" and property[0].text == "InChI":
            drug_InChI = property[1].text
    # groups
    drug_groups = []
    groups = drug.find(html + "groups")
    for group in groups:
        drug_groups.append(group.text)
    # targets
    data_types = ["targets", "enzymes", "carriers", "transporters"]
    drug_targets = []
    for data_type in data_types:
        targets = drug.find(html + data_type)
        for target in targets:
            target_id = target.find(html + "id").text
            polypeptide = target.find(html + "polypeptide")
            if polypeptide == None:
                continue
            target_uniport_id = polypeptide.attrib["id"]
            target_acid_sequence = polypeptide.find(html + "amino-acid-sequence").text
            if target_acid_sequence[0] != ">":
                print("123")
            # polypeptide id and  amino-acid-sequence tag
            temp_dict = {"target_id": target_id, "target_uniport_id": target_uniport_id,
                         "target_acid_sequence": target_acid_sequence}
            drug_targets.append(temp_dict)
            if target_id not in target_ids:
                i += 1
                target_ids.append(target_id)
                # target info
                temp = f">{target_id}|{target_uniport_id}|{target_acid_sequence[1:]}\n"
                write_file(content=temp, file_path="data/target.fasta")
    for target in drug_targets:
        write_csv(drug_id, target["target_id"], file_path="../data/Positive.csv")
    write_csv(drug_id, drug_SMILES, file_path="../data/smiles.csv")
    write_csv(drug_id, drug_InChI, file_path="../data/InChI.csv")

