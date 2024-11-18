-- Functional Query Definition
-- Approved February 1998


select
l_orderkey,
sum(l_extendedprice * (1 - l_discount)) as revenue,
o_orderdate,
o_shippriority
from
customer_1_prt_p21,
orders_1_prt_p21,
lineitem_1_prt_p21
where
c_mktsegment = 'BUILDING'
and c_custkey = o_custkey
and l_orderkey = o_orderkey
and o_orderdate < date '1995-03-15'
and l_shipdate > date '1995-03-15'
group by
l_orderkey,
o_orderdate,
o_shippriority
order by
revenue desc,
o_orderdate;
limit 10;
-- $ID$
-- TPC-H/TPC-R Order Priority Checking 
