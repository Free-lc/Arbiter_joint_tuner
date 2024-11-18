-- Functional Query Definition
-- Approved February 1998


select
c_custkey,
c_name,
sum(l_extendedprice * (1 - l_discount)) as revenue,
c_acctbal,
n_name,
c_address,
c_phone,
c_comment
from
customer_1_prt_p46,
orders_1_prt_p46,
lineitem_1_prt_p46,
nation_1_prt_p46
where
c_custkey = o_custkey
and l_orderkey = o_orderkey
and o_orderdate >= date '1993-10-01'
and o_orderdate < date '1993-10-01' + interval '3' month
and l_returnflag = 'R'
and c_nationkey = n_nationkey
group by
c_custkey,
c_name,
c_acctbal,
c_phone,
n_name,
c_address,
c_comment
order by
revenue desc;
limit 20;
-- $ID$
-- TPC-H/TPC-R Important Stock Identification 
