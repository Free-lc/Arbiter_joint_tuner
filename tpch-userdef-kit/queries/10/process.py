# for i in range(22):
#     input_file = f"/gyc_data/fray_data/Arbiter/tpch-userdef-kit/queries/10_standard/{i+1}_0.sql"
#     with open(input_file, 'r') as file:
#         content = file.read()
#     for j in range(50):
#         if j == 0:
#             continue
#         output_file = f"/gyc_data/fray_data/Arbiter/tpch-userdef-kit/queries/10_standard/{i+1}_{j}.sql"
#         modified_content = content.replace('1_prt_p0', f"1_prt_p{j}")
#         with open(output_file, 'w') as file:
#             file.write(modified_content)
from collections import Counter
import pickle

def find_join_attributes(query):
    lines = query.split("\n")
    join_attributes = []

    for line in lines:
        if "=" in line:
            parts = line.split("=")
            left_attr = parts[0].strip().split(" ")[-1].split(".")[-1]
            right_attr = parts[1].strip().split(" ")[0].split(".")[-1]

            if "_" in left_attr and "_" in right_attr:
                join_attributes.append([left_attr, right_attr])

    return join_attributes

def find_most_frequent(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    if most_common:
        return most_common[0][0]
    else:
        return None
    
for i in range(22):
    input_file = f"/gyc_data/fray_data/Arbiter/tpch-userdef-kit/queries/10/{i+1}.sql"
    with open(input_file, 'r') as file:
        content = file.read()
    join_attributes = find_join_attributes(content)
    join_keys = {}
    for attr_pair in join_attributes:
        for join_pair in attr_pair:
            if "p_" in join_pair:
                if 'part' in join_keys:
                    join_keys['part'] += [join_pair]
                else:
                    join_keys['part'] = [join_pair]
            elif "ps_" in join_pair:
                if 'partsupp' in join_keys:
                    join_keys['partsupp'] += [join_pair]
                else:
                    join_keys['partsupp'] = [join_pair]
            elif "s_" in join_pair:
                if 'supplier' in join_keys:
                    join_keys['supplier'] += [join_pair]
                else:
                    join_keys['supplier'] = [join_pair]
            elif "n_" in join_pair:
                if 'nation' in join_keys:
                    join_keys['nation'] += [join_pair]
                else:
                    join_keys['nation'] = [join_pair]
            elif "r_" in join_pair:
                if 'region' in join_keys:
                    join_keys['region'] += [join_pair]
                else:
                    join_keys['region'] = [join_pair]
            elif "c_" in join_pair:
                if 'customer' in join_keys:
                    join_keys['customer'] += [join_pair]
                else:
                    join_keys['customer'] = [join_pair]
            elif "o_" in join_pair:
                if 'orders' in join_keys:
                    join_keys['orders'] += [join_pair]
                else:
                    join_keys['orders'] = [join_pair]
            elif "l_" in join_pair:
                if 'lineitem' in join_keys:
                    join_keys['lineitem'] += [join_pair]
                else:
                    join_keys['lineitem'] = [join_pair]
            else:
                assert 0, "wrong value"
    for key,value in join_keys.items():
        if len(value) == 0:
            join_keys[key] = None
        join_keys[key] = find_most_frequent(value)
    print(i+1, join_keys)
    with open(f"/gyc_data/fray_data/Arbiter/tpch-userdef-kit/queries/10/{i+1}_join_key.pkl","wb") as f:
        pickle.dump(join_keys, f)