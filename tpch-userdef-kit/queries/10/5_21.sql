-- Functional Query Definition
-- Approved February 1998


select
n_name,
sum(l_extendedprice * (1 - l_discount)) as revenue
from
customer_1_prt_p21,
orders_1_prt_p21,
lineitem_1_prt_p21,
supplier_1_prt_p21,
nation_1_prt_p21,
region_1_prt_p21
where
c_custkey = o_custkey
and l_orderkey = o_orderkey
and l_suppkey = s_suppkey
and c_nationkey = s_nationkey
and s_nationkey = n_nationkey
and n_regionkey = r_regionkey
and r_name = 'ASIA'
and o_orderdate >= date '1994-01-01'
and o_orderdate < date '1994-01-01' + interval '1' year
group by
n_name
order by
revenue desc;
limit -1;
-- $ID$
-- TPC-H/TPC-R Forecasting Revenue Change 
