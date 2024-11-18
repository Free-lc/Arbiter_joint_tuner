# Arbiter: A Joint Tuner for Partitioning and Customized Indexing with Trusted Bayesian Optimization.

## Introduction

Arbiter is a joint index and partition tuning method with Trusted Bayesian Optimization. Arbiter explores distinct index configurations for each data partition,  enabling fine-grained index selection to ensure optimal performance when accessing each local part of the dataset. Furthermore, Arbiter merges the configuration spaces of partition and index into a joint space and performs a state-of-the-art-AI-driven "black box" algorithm namely, Bayesian Optimization (BO) to search for a global partition-index configuration. This ensures effective coordination between the two components, allowing them to synergistically maximize the overall performance. Another contribution in this paper is Trusted Bayesian Optimization (TBO), which integrates two key techniques to enhance traditional Bayesian Optimization. First, TBO effectively compacts the joint configuration space by leveraging prior information from data skipping indexes. Second, it incorporates a white-box module to accelerate the search for promising configurations within the compacted space. Extensive experiments on Greenplum with three benchmark workloads demonstrate thatArbiter can find better configurations by $1.04\times$- $4.42\times$ speedups on the workload execution time compared with the state-of-the-art methods.

## Setup

### Base OS

Ubuntu 20.04

### Software Requirements

python3.8

### Installing Requirements

This demo is based on a modified version of greenplum-db-6.23.4, these are some related installation processes.

```
1. make sure you have installed greenplum database
2. python3.8 -m venv py38venv
3. source py38venv/bin/activate
4. pip install -r requirements.txt   
```

## JSON Configuration files

The experiments and models are configured via JSON files. For examples, check the .json files in the experiments (benchmark_results/fray_op/config_tpch.json). In the following, we explain the different configuration options:

- database_system: the type of databas, postgres is used by default, and greenplum is also represented by postgres.
- benchmark_name: tpch_userdef(TPC-H), tpcds(TPC-DS), wikimedia.
- scale_factor: scale_factor of datasets.
- partition_max: the maximum number of data block of a single table in a specific dataset.
- query_frequence: the maximum frequency of the query in the workload.
- sql_per_table: The maximum number of queries per table。
- budget: the metadata budget of indexes.
- skipping_index: MinMax(zonemap), BloomFilter(bloom filter).

## Start Server

```
python selection/compare.py
```

The program executes ourmethod with SpaceCompactor, ourmethod without SpaceCompactor, and baseline based on the selected json file. Notably, if you are selecting a particular size of data set, you may need to regenerate the required query load, see the generate_queries function of selection/db_openbox.py。

## Running All Experiments

### Overall Performance

`python selection/compare.py`

The program executes ourmethod with SpaceCompactor, ourmethod without SpaceCompactor, and baseline based on the selected json file.

### Impact of Space Compactor.

The code for running Arbiter without Space Compactor can be found in the program selection/compare.py. Use the parameters provided: 'Fray' represents Arbiter without Space Compactor, while 'FrayPreFilter' represents Arbiter with Space Compactor.

```
selection_algorithm = ALGORITHMS["Fray"](black_box, whitebox= False)
index_combination_size,cost = selection_algorithm.get_optimal_value()
```



### Impact of the White-box Model.

The value the parameter 'whitebox' of ALGORITHMS can control the activation and deactivation of the white-box model.

```
# activate the white-box model
selection_algorithm = ALGORITHMS["FrayPreFilter"](black_box, train=False, whitebox= True)
# deactivate the white-box model
selection_algorithm = ALGORITHMS["FrayPreFilter"](black_box, train=False, whitebox= False)
```

 

### Impact of Customized Partition-specific Indexing. 

You need to modify the code in selection/db_openbox.py by yourself; readme to be improved.

### Multi-column indexes Or Single-column indexes. 

You need to modify the code in selection/db_openbox.py by yourself; readme to be improved.

### Extendability Across Data Skipping Indexes.

By modifying the "skipping_index" parameter in the experiment's JSON file to "BloomFilter", you can turn the data skipping index into Bloom filters.

### Block Size Scalability.

By modifying the "partition_max" parameter in the experiment's JSON file to other number, you can change the number of data blocks of each table.

