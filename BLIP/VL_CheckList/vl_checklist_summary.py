import json
import os
import re

# Python code to merge dict using update() method
def Merge(dict1, dict2):
	return(dict2.update(dict1))


# Driver code
dict1 = {'a': 10, 'b': 8}
dict2 = {'d': 6, 'c': 4}

# This returns None
print(Merge(dict1, dict2))

# changes made in dict2
print(dict2)




vl_checklist_total = {"relation": {"sum":0.0, "samples": 0.0}, "attribute": {"sum":0.0, "samples": 0.0}, "object": {"sum":0.0, "samples": 0.0}}
vl_checklist_non_vg = {"relation": {"sum":0.0, "samples": 0.0}, "attribute": {"sum":0.0, "samples": 0.0}, "object": {"sum":0.0, "samples": 0.0}}
vl_checklist_non_vg_with_vaw = {"relation": {"sum":0.0, "samples": 0.0}, "attribute": {"sum":0.0, "samples": 0.0}, "object": {"sum":0.0, "samples": 0.0}}
subclasses = {"attribute_action" :{"sum":0.0, "samples":0},
"attribute_color" :{"sum":0.0, "samples":0},
"attribute_material" :{"sum":0.0, "samples":0},
"attribute_size" :{"sum":0.0, "samples":0},
"attribute_state" :{"sum":0.0, "samples":0},
"object_location" :{"sum":0.0, "samples":0},
"object_size" :{"sum":0.0, "samples":0},
"relation_action" :{"sum":0.0, "samples":0}}
directory = "output/only_negatives9/vlchecklist/5"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if not os.path.isfile(f):
        continue
    json_filename_parts = filename.split(".")
    json_filename_parts = json_filename_parts[0].split("_")
    if "summary" in json_filename_parts:
        continue
    file = open(f)
    json_dict = json.load(file)
    sum = json_dict["total_acc"] * json_dict["number_of_data"]
    samples = json_dict["number_of_data"]
    vl_cl_cat = json_filename_parts[0].lower()
    vl_cl_sub_cat = json_filename_parts[1].lower()
    vl_checklist_total[vl_cl_cat]["sum"] += sum
    vl_checklist_total[vl_cl_cat]["samples"] += samples
    if "swig" in json_filename_parts or "hake" in json_filename_parts:
        vl_checklist_non_vg[vl_cl_cat]["sum"] += sum
        vl_checklist_non_vg[vl_cl_cat]["samples"] += samples
    if "vaw" in json_filename_parts:
        vl_checklist_non_vg_with_vaw[vl_cl_cat]["sum"] += sum
        vl_checklist_non_vg_with_vaw[vl_cl_cat]["samples"] += samples
        vl_checklist_non_vg[vl_cl_cat]["sum"] += sum
        vl_checklist_non_vg[vl_cl_cat]["samples"] += samples
    if "swig" in json_filename_parts or "hake" in json_filename_parts or "vaw" in json_filename_parts:
        dict_name = vl_cl_cat + "_" + vl_cl_sub_cat
        subclasses[dict_name]["sum"] += sum
        subclasses[dict_name]["samples"] += samples



full_summary = {}
full_summary["total"] = {"relation": vl_checklist_total["relation"]["sum"]/vl_checklist_total["relation"]["samples"], "attribute": vl_checklist_total["attribute"]["sum"]/vl_checklist_total["attribute"]["samples"], "object": vl_checklist_total["object"]["sum"]/vl_checklist_total["object"]["samples"]}
full_summary["non_vg"] = {"relation": vl_checklist_non_vg["relation"]["sum"]/vl_checklist_non_vg["relation"]["samples"], "object": vl_checklist_non_vg["object"]["sum"]/vl_checklist_non_vg["object"]["samples"]}
full_summary["non_vg_with_vaw"] = {"attribute": vl_checklist_non_vg_with_vaw["attribute"]["sum"]/vl_checklist_non_vg_with_vaw["attribute"]["samples"]}
subclasses_summary = {k: v["sum"]/v["samples"] for (k,v) in subclasses.items()}
Merge(subclasses_summary,full_summary)
out_file = open(os.path.join(directory,"full_summary.json"), "w") 
json.dump(full_summary, out_file, indent = 6)
out_file.close()