import tensorflow as tf
import numpy as np
import json
import time
ISOTIMEFORMAT="%Y-%m-%d %X"

tf.flags.DEFINE_string("data_dir", "./data", "The data dir.")
tf.flags.DEFINE_string("sub_dir", "WikiPeople", "The sub data dir.")
tf.flags.DEFINE_string("old_dir", "avs", "The dir of the original data representation.")

FLAGS = tf.flags.FLAGS  
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

def write_json(t_t):
    g = open(FLAGS.data_dir+"/"+FLAGS.sub_dir+"/n-ary_"+t_t+".json", "w")
    with open(FLAGS.data_dir+"/"+FLAGS.sub_dir+"/"+FLAGS.old_dir+"/n-ary_"+t_t+".json", "r") as f:
        for line in f:
            aline = ()        
            tmp_dict = eval(line)
            xx_dict = {}
            for k in tmp_dict:
                if k.endswith("_h"):
                    xx_dict["H"] = tmp_dict[k]
                    xx_dict["R"] = k[0:-2]
                elif k.endswith("_t"):
                    xx_dict["T"] = tmp_dict[k]
                elif k != "N":
                    xx_dict[k] = tmp_dict[k]
            xx_dict["N"] = tmp_dict["N"]
            json.dump(xx_dict, g)
            g.write("\n")
    g.close()

if __name__ == "__main__":
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
    arr = ["train", "valid", "test"]
    for i in arr:
        write_json(i)
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
